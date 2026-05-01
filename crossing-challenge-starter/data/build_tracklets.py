#!/usr/bin/env python
"""Build per-frame pedestrian tracklets from JAAD + PIE annotation XMLs.

Inputs (clone these first next to this script):
    data/raw/JAAD/annotations/video_XXXX.xml                 (JAAD, CVAT XML)
    data/raw/PIE/annotations/annotations/setXX/video_XXXX_annt.xml
    data/raw/PIE/annotations/annotations_vehicle/setXX/video_XXXX_obd.xml

Output:
    data/tracklets_raw.parquet  -- one row per (pedestrian, frame) at native 30 fps

The downstream step (build_windows.py) downsamples to 15 Hz and slices into
prediction windows.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent
JAAD_DIR = ROOT / "raw" / "JAAD" / "annotations"
PIE_PED_DIR = ROOT / "raw" / "PIE" / "annotations" / "annotations"
PIE_OBD_DIR = ROOT / "raw" / "PIE" / "annotations" / "annotations_vehicle"


def _ped_attrs(box_el: ET.Element) -> dict[str, str]:
    out: dict[str, str] = {}
    for a in box_el.findall("attribute"):
        out[a.attrib["name"]] = (a.text or "").strip()
    return out


def parse_jaad_video(xml_path: Path) -> list[dict]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    video_name = xml_path.stem  # "video_0001"
    meta = root.find(".//video_attributes")
    tod = meta.findtext("time_of_day", "") if meta is not None else ""
    weather = meta.findtext("weather", "") if meta is not None else ""
    location = meta.findtext("location", "") if meta is not None else ""
    size = root.find(".//original_size")
    fw = int(size.findtext("width", "1920")) if size is not None else 1920
    fh = int(size.findtext("height", "1080")) if size is not None else 1080

    rows: list[dict] = []
    for track in root.findall("track"):
        if track.attrib.get("label") != "pedestrian":
            continue
        for box in track.findall("box"):
            if int(box.attrib.get("outside", "0")) == 1:
                continue
            attrs = _ped_attrs(box)
            raw_id = attrs.get("id", "")
            if not raw_id:
                continue
            rows.append({
                "source": "jaad",
                "ped_id": f"jaad:{video_name}:{raw_id}",
                "video_id": f"jaad:{video_name}",
                "frame": int(box.attrib["frame"]),
                "x1": float(box.attrib["xtl"]),
                "y1": float(box.attrib["ytl"]),
                "x2": float(box.attrib["xbr"]),
                "y2": float(box.attrib["ybr"]),
                "frame_w": fw,
                "frame_h": fh,
                "cross": attrs.get("cross", ""),
                "action": attrs.get("action", ""),
                "occlusion": attrs.get("occlusion", ""),
                "time_of_day": tod,
                "weather": weather,
                "location": location,
                "ego_speed_ms": float("nan"),
                "ego_yaw_rate": float("nan"),
                "ego_heading": float("nan"),
            })
    return rows


def parse_pie_obd(xml_path: Path) -> dict[int, dict[str, float]]:
    """Return {frame_id: {OBD_speed, gyroZ, heading_angle}}."""
    if not xml_path.exists():
        return {}
    tree = ET.parse(xml_path)
    out: dict[int, dict[str, float]] = {}
    for f in tree.getroot().findall("frame"):
        try:
            fid = int(f.attrib["id"])
        except (KeyError, ValueError):
            continue
        out[fid] = {
            "ego_speed_ms": float(f.attrib.get("OBD_speed", "nan")) / 3.6,  # km/h → m/s
            "ego_yaw_rate": float(f.attrib.get("gyroZ", "nan")),
            "ego_heading": float(f.attrib.get("heading_angle", "nan")),
        }
    return out


def parse_pie_video(ped_xml: Path, obd_xml: Path) -> list[dict]:
    tree = ET.parse(ped_xml)
    root = tree.getroot()
    # ped_xml name: set01_video_0001_annt.xml → video_id = pie:set01_video_0001
    vid_stem = ped_xml.stem.replace("_annt", "")
    video_id = f"pie:{vid_stem}"
    size = root.find(".//original_size")
    fw = int(size.findtext("width", "1920")) if size is not None else 1920
    fh = int(size.findtext("height", "1080")) if size is not None else 1080

    obd = parse_pie_obd(obd_xml)

    rows: list[dict] = []
    for track in root.findall("track"):
        if track.attrib.get("label") != "pedestrian":
            continue
        for box in track.findall("box"):
            if int(box.attrib.get("outside", "0")) == 1:
                continue
            attrs = _ped_attrs(box)
            raw_id = attrs.get("id", "")
            if not raw_id:
                continue
            frame = int(box.attrib["frame"])
            obd_row = obd.get(frame, {})
            rows.append({
                "source": "pie",
                "ped_id": f"pie:{vid_stem}:{raw_id}",
                "video_id": video_id,
                "frame": frame,
                "x1": float(box.attrib["xtl"]),
                "y1": float(box.attrib["ytl"]),
                "x2": float(box.attrib["xbr"]),
                "y2": float(box.attrib["ybr"]),
                "frame_w": fw,
                "frame_h": fh,
                "cross": attrs.get("cross", ""),
                "action": attrs.get("action", ""),
                "occlusion": attrs.get("occlusion", ""),
                "time_of_day": "",   # PIE doesn't tag this at the video level in the same slot
                "weather": "",
                "location": "",
                "ego_speed_ms": obd_row.get("ego_speed_ms", float("nan")),
                "ego_yaw_rate": obd_row.get("ego_yaw_rate", float("nan")),
                "ego_heading": obd_row.get("ego_heading", float("nan")),
            })
    return rows


def main() -> None:
    rows: list[dict] = []

    if JAAD_DIR.exists():
        jaad_xmls = sorted(JAAD_DIR.glob("video_*.xml"))
        print(f"JAAD: {len(jaad_xmls)} videos")
        for xml in tqdm(jaad_xmls, desc="jaad"):
            rows.extend(parse_jaad_video(xml))
    else:
        print(f"WARN: {JAAD_DIR} missing — skipping JAAD", file=sys.stderr)

    if PIE_PED_DIR.exists():
        pie_xmls = sorted(PIE_PED_DIR.glob("set*/video_*_annt.xml"))
        print(f"PIE:  {len(pie_xmls)} videos")
        for ped_xml in tqdm(pie_xmls, desc="pie"):
            set_dir = ped_xml.parent.name
            vid_stem = ped_xml.stem.replace("_annt", "")
            obd_xml = PIE_OBD_DIR / set_dir / f"{vid_stem}_obd.xml"
            rows.extend(parse_pie_video(ped_xml, obd_xml))
    else:
        print(f"WARN: {PIE_PED_DIR} missing — skipping PIE", file=sys.stderr)

    df = pd.DataFrame(rows)
    out = ROOT / "tracklets_raw.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df):,} frame-rows → {out}")
    print(f"  pedestrians: {df['ped_id'].nunique():,}")
    print(f"  by source:\n{df.groupby('source')['ped_id'].nunique()}")


if __name__ == "__main__":
    main()
