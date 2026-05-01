"""Trains the intentionally-naive baseline: XGBoost on engineered tracklet
features for intent, plus constant-velocity trajectory (done at predict time).

Writes model.pkl. Run once:

    python baseline.py

This baseline is deliberately weak — it's the bar to beat, not the finish
line.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from xgboost import XGBClassifier, XGBRegressor

from predict import HORIZON_KEYS, _engineered_features, _robust_cv_trajectory, _trajectory_features

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"


REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


def row_to_request(row: pd.Series) -> dict:
    return {k: row[k] for k in REQUEST_FIELDS}


def featurize(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    sample = _engineered_features(row_to_request(df.iloc[0]))
    X = np.empty((n, len(sample)), dtype=np.float32)
    X[0] = sample
    for i in range(1, n):
        X[i] = _engineered_features(row_to_request(df.iloc[i]))
    return X


def tune_intent_calibration(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    probs = np.clip(probs.astype(np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(probs / (1.0 - probs))

    def objective(params: np.ndarray) -> float:
        scale, bias = params
        calibrated = 1.0 / (1.0 + np.exp(-(scale * logits + bias)))
        return log_loss(y_true, np.clip(calibrated, 1e-6, 1.0 - 1e-6))

    result = minimize(objective, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    scale, bias = result.x
    print(f"  calibration scale {scale:.3f}, bias {bias:.3f}, Dev log-loss {result.fun:.4f}")
    return float(scale), float(bias)


def tune_intent_ensemble(prob_matrix: np.ndarray, y_true: np.ndarray) -> tuple[list[float], tuple[float, float]]:
    prob_matrix = np.clip(prob_matrix.astype(np.float64), 1e-6, 1.0 - 1e-6)

    def blend_loss(weights: np.ndarray) -> float:
        blended = np.clip(prob_matrix @ weights, 1e-6, 1.0 - 1e-6)
        return log_loss(y_true, blended)

    n_models = prob_matrix.shape[1]
    result = minimize(
        blend_loss,
        x0=np.full(n_models, 1.0 / n_models),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},),
    )
    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()
    blended = np.clip(prob_matrix @ weights, 1e-6, 1.0 - 1e-6)
    print(f"  intent ensemble weights {weights.round(3).tolist()}, Dev log-loss {result.fun:.4f}")
    calibration = tune_intent_calibration(blended, y_true)
    return weights.astype(float).tolist(), calibration


def prune_intent_models(models: list, weights: list[float], min_weight: float = 1e-4) -> tuple[list, list[float]]:
    kept = [(m, float(w)) for m, w in zip(models, weights) if float(w) > min_weight]
    if not kept:
        return [models[0]], [1.0]
    kept_models, kept_weights = zip(*kept)
    weights_arr = np.asarray(kept_weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()
    return list(kept_models), weights_arr.astype(float).tolist()


def set_inference_jobs(model, n_jobs: int = 1):
    if hasattr(model, "n_jobs"):
        model.n_jobs = n_jobs
    return model


def featurize_trajectory(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    sample = _trajectory_features(row_to_request(df.iloc[0]))
    X = np.empty((n, len(sample)), dtype=np.float32)
    X[0] = sample
    for i in range(1, n):
        X[i] = _trajectory_features(row_to_request(df.iloc[i]))
    return X


def trajectory_targets(df: pd.DataFrame) -> np.ndarray:
    y = np.empty((len(df), len(HORIZON_KEYS) * 2), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        hist = np.stack([np.asarray(r, dtype=np.float64) for r in row.bbox_history])
        cur_cx = (hist[-1, 0] + hist[-1, 2]) * 0.5
        cur_cy = (hist[-1, 1] + hist[-1, 3]) * 0.5
        fw = float(row.frame_w)
        fh = float(row.frame_h)
        vals = []
        for key in HORIZON_KEYS:
            box = np.asarray(getattr(row, key), dtype=np.float64)
            tcx = (box[0] + box[2]) * 0.5
            tcy = (box[1] + box[3]) * 0.5
            vals.extend([(tcx - cur_cx) / fw, (tcy - cur_cy) / fh])
        y[i] = vals
    return y


def robust_cv_offsets(df: pd.DataFrame) -> np.ndarray:
    y = np.empty((len(df), len(HORIZON_KEYS) * 2), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        row_dict = row._asdict()
        req = row_to_request(pd.Series(row_dict))
        hist = np.stack([np.asarray(r, dtype=np.float64) for r in row.bbox_history])
        cur_cx = (hist[-1, 0] + hist[-1, 2]) * 0.5
        cur_cy = (hist[-1, 1] + hist[-1, 3]) * 0.5
        fw = float(row.frame_w)
        fh = float(row.frame_h)
        cv = _robust_cv_trajectory(req)
        vals = []
        for key in HORIZON_KEYS:
            box = np.asarray(cv[key], dtype=np.float64)
            pcx = (box[0] + box[2]) * 0.5
            pcy = (box[1] + box[3]) * 0.5
            vals.extend([(pcx - cur_cx) / fw, (pcy - cur_cy) / fh])
        y[i] = vals
    return y


def centers_from_boxes(boxes: list[list[float]]) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float64)
    return np.column_stack(((arr[:, 0] + arr[:, 2]) * 0.5, (arr[:, 1] + arr[:, 3]) * 0.5))


def tune_trajectory_blend(dev: pd.DataFrame, learned_offsets: np.ndarray) -> list[float]:
    blends: list[float] = []
    alphas = np.linspace(0.0, 1.0, 21)

    for h_idx, key in enumerate(HORIZON_KEYS):
        learned_centers = np.empty((len(dev), 2), dtype=np.float64)
        cv_centers = np.empty((len(dev), 2), dtype=np.float64)
        true_centers = np.empty((len(dev), 2), dtype=np.float64)

        for i, row in enumerate(dev.itertuples(index=False)):
            req = row_to_request(pd.Series(row._asdict()))
            hist = np.stack([np.asarray(r, dtype=np.float64) for r in row.bbox_history])
            cur_cx = (hist[-1, 0] + hist[-1, 2]) * 0.5
            cur_cy = (hist[-1, 1] + hist[-1, 3]) * 0.5
            learned_centers[i, 0] = cur_cx + learned_offsets[i, 2 * h_idx] * float(row.frame_w)
            learned_centers[i, 1] = cur_cy + learned_offsets[i, 2 * h_idx + 1] * float(row.frame_h)

            cv_box = _robust_cv_trajectory(req)[key]
            cv_centers[i] = [(cv_box[0] + cv_box[2]) * 0.5, (cv_box[1] + cv_box[3]) * 0.5]

            true_box = np.asarray(getattr(row, key), dtype=np.float64)
            true_centers[i] = [(true_box[0] + true_box[2]) * 0.5, (true_box[1] + true_box[3]) * 0.5]

        best_alpha = 0.0
        best_ade = float("inf")
        for alpha in alphas:
            pred_centers = alpha * learned_centers + (1.0 - alpha) * cv_centers
            ade = float(np.hypot(pred_centers[:, 0] - true_centers[:, 0], pred_centers[:, 1] - true_centers[:, 1]).mean())
            if ade < best_ade:
                best_ade = ade
                best_alpha = float(alpha)
        blends.append(best_alpha)
        print(f"  {key}: blend {best_alpha:.2f}, Dev ADE {best_ade:.2f}px")

    return blends


def tune_trajectory_ensemble(
    dev: pd.DataFrame,
    direct_offsets: np.ndarray,
    residual_offsets: np.ndarray,
    extra_offsets: np.ndarray,
) -> list[list[float]]:
    weights: list[list[float]] = []
    cv_offsets = robust_cv_offsets(dev)
    grid = np.linspace(0.0, 1.0, 21)

    for h_idx, key in enumerate(HORIZON_KEYS):
        direct_centers = np.empty((len(dev), 2), dtype=np.float64)
        residual_centers = np.empty((len(dev), 2), dtype=np.float64)
        extra_centers = np.empty((len(dev), 2), dtype=np.float64)
        cv_centers = np.empty((len(dev), 2), dtype=np.float64)
        true_centers = np.empty((len(dev), 2), dtype=np.float64)

        for i, row in enumerate(dev.itertuples(index=False)):
            hist = np.stack([np.asarray(r, dtype=np.float64) for r in row.bbox_history])
            cur_cx = (hist[-1, 0] + hist[-1, 2]) * 0.5
            cur_cy = (hist[-1, 1] + hist[-1, 3]) * 0.5
            fw = float(row.frame_w)
            fh = float(row.frame_h)

            cv_centers[i, 0] = cur_cx + cv_offsets[i, 2 * h_idx] * fw
            cv_centers[i, 1] = cur_cy + cv_offsets[i, 2 * h_idx + 1] * fh
            direct_centers[i, 0] = cur_cx + direct_offsets[i, 2 * h_idx] * fw
            direct_centers[i, 1] = cur_cy + direct_offsets[i, 2 * h_idx + 1] * fh
            residual_centers[i, 0] = cv_centers[i, 0] + residual_offsets[i, 2 * h_idx] * fw
            residual_centers[i, 1] = cv_centers[i, 1] + residual_offsets[i, 2 * h_idx + 1] * fh
            extra_centers[i, 0] = cur_cx + extra_offsets[i, 2 * h_idx] * fw
            extra_centers[i, 1] = cur_cy + extra_offsets[i, 2 * h_idx + 1] * fh

            true_box = np.asarray(getattr(row, key), dtype=np.float64)
            true_centers[i] = [(true_box[0] + true_box[2]) * 0.5, (true_box[1] + true_box[3]) * 0.5]

        best_w = [0.0, 0.0, 0.0, 1.0]
        best_ade = float("inf")
        for w_direct in grid:
            for w_residual in grid:
                for w_extra in grid:
                    if w_direct + w_residual + w_extra > 1.0:
                        continue
                    w_cv = 1.0 - w_direct - w_residual - w_extra
                    pred_centers = (
                        w_direct * direct_centers
                        + w_residual * residual_centers
                        + w_cv * cv_centers
                        + w_extra * extra_centers
                    )
                    ade = float(np.hypot(pred_centers[:, 0] - true_centers[:, 0], pred_centers[:, 1] - true_centers[:, 1]).mean())
                    if ade < best_ade:
                        best_ade = ade
                        best_w = [float(w_direct), float(w_residual), float(w_cv), float(w_extra)]
        weights.append(best_w)
        print(
            f"  {key}: direct {best_w[0]:.2f}, residual {best_w[1]:.2f}, "
            f"cv {best_w[2]:.2f}, extra {best_w[3]:.2f}, Dev ADE {best_ade:.2f}px"
        )

    return weights


def main() -> None:
    print("Loading train + dev...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")
    print(f"  train: {len(train):,}   dev: {len(dev):,}")
    print(f"  positive rates: train {train.will_cross_2s.mean():.3f}, "
          f"dev {dev.will_cross_2s.mean():.3f}")

    print("\nFeaturizing...")
    t0 = time.time()
    X_train = featurize(train)
    X_dev = featurize(dev)
    y_train = train["will_cross_2s"].to_numpy(dtype=np.int32)
    y_dev = dev["will_cross_2s"].to_numpy(dtype=np.int32)
    print(f"  {time.time() - t0:.1f}s  feature shape: {X_train.shape}")

    pos_ratio = float(y_train.mean())

    print("\nTraining intent classifiers (no class rebalancing — want calibrated probs)...")
    t0 = time.time()
    clf = XGBClassifier(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=0.2,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",
        random_state=2024,
    )
    clf.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=False)

    clf_deep = XGBClassifier(
        n_estimators=260,
        max_depth=5,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_lambda=8.0,
        reg_alpha=0.5,
        min_child_weight=5,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",
        random_state=2025,
    )
    clf_deep.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=False)

    clf_hgb = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.035,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.05,
        random_state=2026,
        early_stopping=True,
        validation_fraction=0.12,
    )
    clf_hgb.fit(X_train, y_train)

    clf_extra = ExtraTreesClassifier(
        n_estimators=160,
        max_depth=16,
        min_samples_leaf=20,
        max_features=0.7,
        random_state=2027,
        n_jobs=-1,
    )
    clf_extra.fit(X_train, y_train)
    print(f"  {time.time() - t0:.1f}s")

    intent_models = [clf, clf_deep, clf_hgb, clf_extra]
    dev_prob_matrix = np.column_stack([m.predict_proba(X_dev)[:, 1] for m in intent_models])
    dev_probs = dev_prob_matrix[:, 0]
    ll = log_loss(y_dev, np.clip(dev_probs, 1e-6, 1 - 1e-6))
    prior_ll = log_loss(y_dev, np.full_like(dev_probs, pos_ratio))
    print(f"\nDev log-loss:  {ll:.4f}  (class-prior baseline {prior_ll:.4f})")
    intent_weights, intent_calibration = tune_intent_ensemble(dev_prob_matrix, y_dev)
    intent_models, intent_weights = prune_intent_models(intent_models, intent_weights)
    for model in intent_models:
        set_inference_jobs(model)

    print("\nFeaturizing trajectory...")
    t0 = time.time()
    XT_train = featurize_trajectory(train)
    XT_dev = featurize_trajectory(dev)
    yt_train = trajectory_targets(train)
    cv_train = robust_cv_offsets(train)
    print(f"  {time.time() - t0:.1f}s  trajectory feature shape: {XT_train.shape}")

    print("\nTraining direct trajectory regressors...")
    t0 = time.time()
    trajectory = []
    dev_offsets = np.empty((len(dev), yt_train.shape[1]), dtype=np.float32)
    for target_idx in range(yt_train.shape[1]):
        reg = XGBRegressor(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=2024 + target_idx,
        )
        reg.fit(XT_train, yt_train[:, target_idx], verbose=False)
        set_inference_jobs(reg)
        dev_offsets[:, target_idx] = reg.predict(XT_dev)
        trajectory.append(reg)
    print(f"  {time.time() - t0:.1f}s")

    print("\nTraining residual trajectory regressors...")
    t0 = time.time()
    trajectory_residual = []
    residual_targets = yt_train - cv_train
    dev_residual_offsets = np.empty((len(dev), yt_train.shape[1]), dtype=np.float32)
    for target_idx in range(yt_train.shape[1]):
        reg = XGBRegressor(
            n_estimators=260,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.5,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=3024 + target_idx,
        )
        reg.fit(XT_train, residual_targets[:, target_idx], verbose=False)
        set_inference_jobs(reg)
        dev_residual_offsets[:, target_idx] = reg.predict(XT_dev)
        trajectory_residual.append(reg)
    print(f"  {time.time() - t0:.1f}s")

    print("\nTraining extra-trees trajectory regressor...")
    t0 = time.time()
    trajectory_extra = ExtraTreesRegressor(
        n_estimators=80,
        max_depth=22,
        min_samples_leaf=4,
        max_features=0.85,
        random_state=2024,
        n_jobs=-1,
    )
    trajectory_extra.fit(XT_train, yt_train)
    set_inference_jobs(trajectory_extra)
    dev_extra_offsets = trajectory_extra.predict(XT_dev)
    print(f"  {time.time() - t0:.1f}s")

    print("\nTuning trajectory ensemble on Dev...")
    trajectory_ensemble = tune_trajectory_ensemble(dev, dev_offsets, dev_residual_offsets, dev_extra_offsets)

    print(f"\nSaving model → {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "intent": intent_models[0],
            "intent_models": intent_models,
            "intent_weights": intent_weights,
            "intent_calibration": intent_calibration,
            "trajectory": trajectory,
            "trajectory_residual": trajectory_residual,
            "trajectory_extra": trajectory_extra,
            "trajectory_ensemble": trajectory_ensemble,
        }, f)


if __name__ == "__main__":
    main()
    sys.exit(0)
