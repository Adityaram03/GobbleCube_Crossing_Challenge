# The ETA Challenge

*A Gobblecube take-home: build the ride-hailing ETA engine.*

---

## The Problem

You've just joined the forecasting team at a ride-hailing company.

Every time a rider opens the app, they see an ETA: *"Your driver arrives in 4 minutes. Trip takes 18 minutes."* That number is the difference between a happy rider and a cancelled trip. It's the difference between a driver earning and a driver idling. Every second of prediction error, at scale, costs real money.

Your job: given a ride request and a year of historical trips, predict the ride duration as accurately as you can.

This is a problem real companies pay serious money to solve. We want to see how you approach it.

---

## What You Ship

A Python function:

```python
def predict(request: dict) -> float:
    """
    Input:  {
        "pickup_zone":     int,   # NYC taxi zone, 1-265
        "dropoff_zone":    int,
        "requested_at":    str,   # ISO 8601 datetime
        "passenger_count": int
    }
    Output: predicted trip duration in seconds (float)
    """
```

Packaged as a GitHub repo containing:

- `predict.py` exposing the function above
- A `Dockerfile` that builds your submission in under 10 minutes
- Your trained model weights (`model.pkl` or equivalent)
- A `README.md` describing your approach

**Constraints:**

- Inference ≤ 200 ms per request on CPU
- Total Docker image ≤ 2.5 GB (the reference baseline builds to ~2.02 GB because xgboost pulls scipy/nccl)
- No external API calls at inference time

---

## The Data

We use **real public NYC Yellow Taxi trip records**.

- **Train**: 11.5 months of 2023 (~37M trips after cleaning)
- **Dev**: last 2 weeks of 2023 (~1M trips, for your local scoring)
- **Eval**: 50k trips from a held-out 2024 slice. Kept by us, never distributed. This is what we grade on.

You download Train and Dev via `python data/download_data.py`. The schema lives in `data/schema.md`.

---

## Scoring

**Your score = Mean Absolute Error (MAE), in seconds, on the held-out Eval set.**

Lower is better. We run your Docker image in a sandbox, stream eval requests through it, and compute MAE. One number, one leaderboard, no subjective judging.

For reference, measured on the Dev set:

| Approach | Dev MAE |
|---|---|
| Predict the global mean | ~580 s |
| Zone-pair averages (10 lines, no ML) | ~300 s |
| **GBT baseline (this repo, intentionally naive)** | **~350 s** |

Use Dev as a self-check. Baseline scores ~351 s on Dev and ~367 s on
Eval, a ~15 s gap because Eval is a held-out winter-holiday slice that
skews slightly harder than Dev. Budget for a similar small loss when we
grade on Eval. Beat the baseline by as much as you can. There's no
posted target; we'll tell you how your number stacks up.

You may notice the naive GBT baseline actually loses to a 10-line zone-pair
lookup. That's not a bug in the starter; it's the first thing worth
understanding before you start training anything fancy.

---

## Submission

- Send us the repo URL when your submission is something you'd put
  your name on.
- One submission per candidate. Do not share work with other candidates.

---

## Rules

- Use any open-source model, library, or dataset you like
- Use any paid LLM API (Claude, GPT, Gemini) during development. We encourage it.
- Your **final submission must not** make external API calls at inference time
- Do not use the 2024 eval set anywhere in training
- Do not use proprietary data from any ride-hailing company

---

## Baseline Submission

```bash
git clone https://github.com/gobblecube-hiring/arena
cd arena/eta-challenge-starter

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ~500 MB download, one-time
python data/download_data.py

# Trains in ~5 min on a laptop CPU, writes model.pkl
python baseline.py

# Scores on Dev
python grade.py
```

What's in the repo:

- `baseline.py`: a gradient-boosted tree on 6 features (pickup zone, dropoff zone, hour of day, day of week, month, passenger count). This is the bar to beat.
- `predict.py`: the submission interface. Signature is fixed; internals are yours.
- `grade.py`: the exact scoring logic we run. Validate locally before submitting.
- `data/download_data.py`: fetches NYC TLC data and builds train/dev splits.
- `Dockerfile`: reference build. Extend as needed.
- `tests/test_submission.py`: smoke tests for the submission contract.

Your job is to ship something better than `baseline.py`.

---

## FAQ

**Is this hard?**
Yes. The baseline is deliberately honest. Meaningfully beating it takes focus and good engineering judgment.

**What AI tooling should I use?**
Whatever helps you ship. Claude Code is our in-house default and the
fastest path we've seen on these challenges, but Cursor, Aider, Copilot,
ChatGPT, direct API calls, or no LLM at all are all fine. The role is
about shipping fast with AI pair-programming generally, not about any
one tool. Your git history is part of the signal. We read commits, not
just the final state. Real iteration with AI help looks different from a
polished from-scratch dump.

**I've never trained a deep learning model. Should I apply?**
Yes. The baseline is CPU-only and you can beat it without a neural network. Solid feature engineering and a tabular model will get you a meaningful improvement. The strongest approaches to this type of problem typically use deep learning, though, so the strongest entries usually pick it up along the way. That's the point of the challenge.

**What if I don't have access to a GPU?**
You can submit a valid entry without one. The baseline is CPU-only. Strong submissions may require GPU training, though, and free-tier notebook environments (Kaggle, Colab, Lightning.ai) each offer ~30 GPU-hours per week, which has typically been enough. More detail in §6.3.

**What do you actually care about?**
Your final score, your README, and your git log, roughly in that order. A clean submission with a thoughtful write-up will beat a slightly better score with no explanation.

**Can I collaborate with a friend?**
No. Individual submissions only.

**What order do I run things in?**

1. `python data/download_data.py` (one-time, ~500 MB)
2. `python baseline.py` (produces `model.pkl`)
3. `python grade.py` (validates on Dev, prints MAE)
4. `docker build -t my-eta .` (packages for submission)
5. `docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv` (test grader pathway)

---

## 6. Resources

### 6.1 Libraries we've seen work well
`pandas`, `numpy`, `polars`, `scikit-learn`, `xgboost`, `lightgbm`, `torch`, `transformers`, `pytorch-lightning`, `geopandas`, `osrm-backend`.

### 6.2 Datasets you may find useful
- NYC TLC trip records (included via `download_data.py`)
- NYC taxi zone shapefile (for centroid coords): https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip
- NOAA hourly weather for JFK/LGA/NYC: https://www.ncei.noaa.gov/access/services/data/v1
- OpenStreetMap road network for NYC (free)

### 6.3 Compute

The baseline trains in about 5 minutes on a laptop CPU. You do not need a GPU to submit a valid entry.

Deep-learning approaches benefit substantially from GPU training.

If you don't have one: free-tier notebook environments (Kaggle, Colab, Lightning.ai) each offer ~30 GPU-hours per week, which has typically been enough compute. How you use that compute is part of the test.

### 6.4 Things that will disqualify you
- Using the 2024 eval set during training
- Submitting something that does not run in our sandbox
- Hardcoding per-request predictions (we fuzz requests)
- Scraping a real ride-hailing company's API

---

## What We Actually Care About

We are **not** hiring a specialist ride-hailing ML engineer. We don't run a ride-hailing company. We use this problem because it's well-defined, has open data, and is outside our business.

What we're hiring for: **an engineer who can pick up a problem they've never seen before, pair effectively with modern AI tooling, and ship something that works.** The ETA problem is just an excuse to watch you do that.

Your submission tells us three things:

1. **Do you ship?** The number on the leaderboard.
2. **Can you learn fast?** Your git log shows the trajectory. First commits usually look nothing like final commits.
3. **Can you reason about a problem that isn't handed to you as a spec?** Your README explains what you tried, what failed, and what the next experiment would be if you kept going.

You don't need an ML background to do well here. The difference is rarely background. It's almost always mindset.

Good luck. We're excited to see what you ship.

---

*Submit your repo URL and LinkedIn profile to agentic-hiring@gobblecube.ai. Questions welcome at the same address.*
