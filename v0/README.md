# v0 MVP Game Recommender

This folder contains the cleaned MVP version of the Steam game recommender system.

## Project layout

- `src/` contains the modular pipeline code.
- `src/main.py` is the executable entrypoint for v0.
- `src/pipeline.py` orchestrates the pipeline stages.
- `notebooks/v0_clean.ipynb` is the stabilized MVP notebook used as the reference workflow.
- `data_clean/raw_clean/` contains the raw input dataset files.
- `data_clean/processed_clean/` contains generated MVP artifacts such as item vectors, train/test splits, and user profiles.
- `data/` and `notebooks/v0.ipynb` are earlier v0 experimental assets kept for traceability.

## Run

From the repository root:

```bash
python -m v0.src.main --stage recommend
```

Useful stages:

- `processed` rebuilds processed item vectors and sampled interactions from raw data.
- `users` rebuilds temporal train/test splits and user profiles from processed artifacts.
- `recommend` runs recommendation, evaluation, and cold-start from existing processed artifacts.
- `full` runs the full v0 pipeline from raw data through evaluation.

## Current MVP flow

Raw data is cleaned, item vectors are built from game metadata and text features, active users are split temporally into train/test histories, user profiles are built from train interactions, candidates are retrieved by cosine similarity, seen items are filtered, candidates are reranked, and held-out positive future interactions are used for offline evaluation.

## Current limitations

- User representations are centroid-based weighted averages of interacted item vectors.
- The system currently uses content-based retrieval and does not yet learn collaborative latent representations.
- Recommendations are evaluated offline using simulated active-user interactions.
- ANN retrieval, diversity reranking, and online serving are not yet implemented in v0.
