# v1 Pipeline — Complete Reference

> **Note:** This document is a detailed historical/conceptual reference for the v1 design.
> The live modular code at `v1/src/` may have diverged in minor ways (signatures, docstrings,
> bug fixes) from what is described here. For exact API contracts, consult the source directly.
> This doc remains valuable for understanding the architecture, data flow, and design rationale.

> Every concept, function, design decision, and data-flow connection in the v1 two-tower game recommendation system, organized by pipeline stage.
> **14 source files · 14 modules · 76 concepts** (incl. 3 ablations + HYBRID_ALPHA sweep)

---

## Variable Renames (clean code pass)

| Old name | New name | Reason |
|---|---|---|
| `full_catalog_games` | `full_games` | Shorter; catalog is implicit |
| `full_catalog_games_metadata` | `full_games_meta` | Shorter |
| `full_catalog_item_features` | `full_item_features` | Shorter |
| `fallback_catalog_games` | `fallback_games` | Shorter |
| `warm_catalog_games` | `warm_games` | Shorter |
| `warm_catalog_games_metadata` | `warm_games_meta` | Shorter |
| `train_simu_users` | `train_interactions` | Were interaction DataFrames, not user lists |
| `val_simu_users` | `val_interactions` | Same as above |
| `test_simu_users` | `test_interactions` | Same as above |
| `full_train_simu_users` | `full_train` | Pre-filter full split |
| `full_val_simu_users` | `full_val` | Pre-filter full split |
| `full_test_simu_users` | `full_test` | Pre-filter full split |
| `final_games_transformed` | `item_features_df` | Describes what it is: item feature DataFrame |
| `user_positive_matrix` | `train_pos_mask` | Shorter; clarifies it's a boolean mask |

---

## Pipeline Architecture

```
                                    ┌────────────────────────┐
                                    │     RAW DATA LOADING    │
                                    │  games.csv, metadata,   │
                                    │  users.csv, interactions│
                                    └───────────┬────────────┘
                                                │
                                    ┌───────────▼────────────┐
                                    │   USER RESERVOIR SAMPLE │
                                    │  3,000 active users     │
                                    │  (≥20 reviews each)     │
                                    └───────────┬────────────┘
                                                │
                        ┌───────────────────────┼───────────────────────┐
                        │                       │                       │
              ┌─────────▼─────────┐  ┌──────────▼──────────┐  ┌─────────▼─────────┐
              │  FEATURE ENGR.    │  │  INTERACTION PREP   │  │  CATALOG PREP     │
              │  (on games)       │  │  dedup, log1p,      │  │  intersect games   │
              │  booleans,rating, │  │  sort, is_positive  │  │  & metadata        │
              │  log1p,pos_ratio  │  └──────────┬──────────┘  └─────────┬─────────┘
              └─────────┬─────────┘             │                       │
                        │                       │                       │
              ┌─────────▼─────────┐             │                       │
              │  TEXT CLEANING    │             │                       │
              │  HTML,URLs,punct  │             │                       │
              └─────────┬─────────┘             │                       │
                        │                       │                       │
              ┌─────────▼─────────┐             │                       │
              │  EMBEDDINGS       │             │                       │
              │  MiniLM (tags)    │             │                       │
              │  MPNet (descs)    │             │                       │
              └─────────┬─────────┘             │                       │
                        │                       │                       │
              ┌─────────▼─────────┐             │                       │
              │  BLOCK MERGE +    │             │                       │
              │  WEIGHT + L2-NORM │             │                       │
              │  → item_features  │             │                       │
              └─────────┬─────────┘             │                       │
                        │                       │                       │
                        │         ┌─────────────▼─────────────┐         │
                        │         │   TEMPORAL SPLIT (70/15/15)│         │
                        │         │   per-user chronological   │         │
                        │         └─────────────┬─────────────┘         │
                        │                       │                       │
                        │         ┌─────────────▼─────────────┐         │
                        │         │  ENGAGEMENT STRENGTH      │         │
                        │         │  clip@99%, MinMax scale,   │         │
                        │         │  weighted blend→impl_rating│         │
                        │         └─────────────┬─────────────┘         │
                        │                       │                       │
                        │         ┌─────────────▼─────────────┐         │
                        │         │  COLLAB. FILTER (iterative)│         │
                        │         │  items≥5 pos, users≥5 ev  │         │
                        │         │  users≥2 pos → warm set   │         │
                        │         └─────────────┬─────────────┘         │
                        │                       │                       │
                        └───────────┬───────────┘                       │
                                    │                                   │
                          ┌─────────▼─────────┐                         │
                          │  CONTIGUOUS INDICES│                         │
                          │  factorize user_id │                         │
                          │  & app_id→0..N-1  │                         │
                          └─────────┬─────────┘                         │
                                    │                                   │
                    ┌───────────────┼───────────────┐                   │
                    │               │               │                   │
          ┌─────────▼─────────┐ ┌──▼──────────┐ ┌──▼──────────────┐    │
          │ positive_history  │ │ seen_item   │ │ heldout_positive│    │
          │ _map (train only) │ │ _map        │ │ _map            │    │
          └─────────┬─────────┘ └──┬──────────┘ └──┬──────────────┘    │
                    │               │               │                   │
          ┌─────────▼─────────┐     │         ┌─────▼──────────────┐   │
          │ build_prefix      │     │         │ filter_targets_by  │   │
          │ _examples         │     │         │ _release_date      │   │
          └─────────┬─────────┘     │         └────────────────────┘   │
                    │               │                                   │
          ┌─────────▼─────────┐     │                                   │
          │ SteamSequence     │     │                                   │
          │ TwoTowerDataset   │     │                                   │
          └─────────┬─────────┘     │                                   │
                    │               │                                   │
          ┌─────────▼─────────┐     │                                   │
          │ DataLoader(batch) │     │                                   │
          └─────────┬─────────┘     │                                   │
                    │               │                                   │
                    └───────┬───────┘                                   │
                            │                                           │
                  ┌─────────▼─────────┐                                 │
                  │  TwoTowerModel    │                                 │
                  │  user_tower       │                                 │
                  │  item_tower       │                                 │
                  └─────────┬─────────┘                                 │
                            │                                           │
                  ┌─────────▼─────────┐                                 │
                  │  train_epoch()    │                                 │
                  │  InfoNCE +        │                                 │
                  │  false-neg mask + │                                 │
                  │  logQ + temp      │                                 │
                  └─────────┬─────────┘                                 │
                            │                                           │
            ┌───────────────┼───────────────┐                           │
            │               │               │                           │
  ┌─────────▼─────────┐ ┌──▼──────────┐ ┌──▼──────────────────┐        │
  │ evaluate_filtered │ │ popularity  │ │ random-expected +    │        │
  │ (val + test)      │ │ baseline    │ │ content-centroid     │        │
  │ Recall@10 + hybrid│ │             │ │ baselines            │        │
  └───────────────────┘ └─────────────┘ └──────────────────────┘        │
                                                                        │
  ┌──────────────────────────────────────────────────────────────────┐  │
  │  COLD-START: recommend_cold_start_v1(fallback_games)             │  │
  └──────────────────────────────────────────────────────────────────┘  │
```

**Core principle: train on the past, evaluate on the future, with zero leakage.**

---

## Stage 1: Data Ingestion & User Sampling

### What happens here

The raw data is too large to hold in memory. We use reservoir sampling to select 3,000 active users, then load only their interactions. This avoids row-sampling interactions (which would break temporal sequences) while keeping the pipeline memory-efficient.

**Cells:** 0–3

**Key parameters:**

| Parameter | Value | Meaning |
|---|---|---|
| `TARGET_SAMPLED_ACTIVE_USERS` | 3,000 | Users to sample |
| `MIN_RAW_REVIEWS` | 20 | Minimum reviews for a user to be "active" |
| `USER_CHUNK_SIZE` | 500,000 | Users processed per chunk |
| `RANDOM_SEED` | 42 | Fixed for reproducibility |

### Concepts in this stage

| # | Concept | What it is |
|---|---|---|
| 1 | **Reservoir sampling** | Uniform random sampling from a streaming CSV (too large for memory). Each user has equal probability of selection regardless of position in the file. |
| 2 | **Active user filtering** (`MIN_RAW_REVIEWS = 20`) | Only users with ≥20 reviews are eligible. Sparse users produce unreliable embeddings. |
| 3 | **Catalog intersection** | Only items present in *both* `games.csv` and `games_metadata.json` survive. Any item in one but not the other is dropped. |

No dedicated functions in this stage — it's inline Python/markdown cells.

---

## Stage 2: Feature Engineering

### What happens here

Games metadata is transformed into a 1,173-dimensional feature vector suitable for the item tower. The pipeline processes five distinct signal families and combines them with learned weights.

### Functions

#### `fit_engagement_transform(train_frame)`
**Cell:** 23

**Signature:**
```python
def fit_engagement_transform(train_frame: pd.DataFrame) -> tuple[dict, MinMaxScaler]
```

**Why it matters:** Engagement metrics (hours played, helpful votes, funny votes) are heavy-tailed. A handful of outliers can dominate scaling. Fitting clipping and scaling on train only prevents information leakage from val/test.

**What it does:** Computes 99th percentile clipping thresholds for `hours_log`, `helpful_log`, and `funny_log` columns using only the training set, then fits `MinMaxScaler` on clipped values. Returns `(clip_upper_dict, fitted_scaler)`.

---

#### `add_engagement_strength(frame, clip_upper, scaler)`
**Cell:** 23

**Signature:**
```python
def add_engagement_strength(frame: pd.DataFrame, clip_upper: dict, scaler: MinMaxScaler) -> pd.DataFrame
```

**Why it matters:** Raw engagement metrics need to be combined into a single signal. The weighted blend (hours=0.75, helpful=0.15, funny=0.10) reflects that playtime is the strongest proxy for enjoyment. Negative interactions get `implicit_rating = 0` since disliking a game carries a qualitatively different signal.

**What it does:**
1. Clips engagement log-columns to train-fit upper bounds
2. Applies `MinMaxScaler` (fitted on train)
3. Computes `implicit_rating = 0.75 × hours + 0.15 × helpful + 0.10 × funny`
4. Zeros out for non-positive (`is_positive == 0`) interactions
5. Returns the DataFrame with an added `implicit_rating` column

### Concepts in this stage

| # | Concept | What it is |
|---|---|---|
| 5 | **Log1p transform for skewed features** | `np.log1p()` on `hours`, `helpful`, `funny` (engagement) and `user_reviews`, `price_final` (metadata). Handles right-skew with zero-inflation. |
| 6 | **Train-only scaling** | Clipping + `MinMaxScaler` fitted on train only, applied to val/test — no data leakage. |
| 7 | **Implicit rating** | `0.75·hours_log + 0.15·helpful_log + 0.10·funny_log`. A confidence/strength signal (not the retrieval label). Currently computed but unused by the two-tower.
| 8 | **Rating one-hot encoding** | `pd.get_dummies` on `rating` → 9 binary features (Mixed, Mostly Positive, Overwhelmingly Positive, etc.). |
| 9 | **Boolean → int8 conversion** | `win`, `mac`, `linux`, `steam_deck` cast from bool to int8. |
| 10 | **positive_ratio normalization** | Divided by 100 to bring percentage into [0, 1] range. |
| 11 | **Text enrichment** | Title merged into description before embedding (e.g., "Enter the dark underworld... Prince of Persia: Warrior Within™"). |
| 12 | **Text cleaning pipeline** | Strip HTML tags, remove URLs, remove punctuation, collapse whitespace — applied separately to tags and description+title. |
| 13 | **MiniLM tag embeddings** (384-dim) | `all-MiniLM-L12-v2` — compact SentenceTransformer for short tag strings. L2-normalized. Missing tags → zero vector. |
| 14 | **MPNet description embeddings** (768-dim) | `all-mpnet-base-v2` — richer SentenceTransformer for description+title text. L2-normalized. Missing descriptions → zero vector. |
| 15 | **Metadata availability flags** | `has_description`, `has_tags`, `has_text_metadata` — binary flags so the item tower can distinguish genuinely missing text from a zero embedding. |
| 16 | **Block-wise item feature construction** | Structured metadata (18 cols) + tags (384 cols) + descriptions (768 cols) + availability flags (3 cols) = 1,173-dim vector. Blocks merged independently to prevent any single family from dominating. |
| 17 | **Item feature table alignment** | Strict `validate="one_to_one"` merges. Assert that `app_id` sets match across all tables before training. |

### Connection: how feature engineering feeds the model

1. The item tower receives `item_features_df` — a pre-computed, L2-normalized 1,173-dim matrix
2. The tower concatenates a learnable `item_id_embedding` (128-dim) with these static features, then projects to the shared 128-dim embedding space
3. For the user tower: history items are looked up by their `item_id_embedding` (NOT the full 1,173-dim feature — the attention mechanism operates in the compact 128-dim space)
4. The weighted block structure (0.15/0.15/0.25/0.45) ensures each signal family contributes proportionally

---

## Stage 3: Temporal Splitting & Leakage Prevention

### What happens here

The interaction data is split into train/val/test per user, by time. This is the most critical architectural decision: all splits respect chronological order so the model can only train on the past and evaluate on the future.

### Functions

#### `temporal_split_by_user(frame, train_ratio, val_ratio)`
**Cell:** 22

**Signature:**
```python
def temporal_split_by_user(
    frame: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
```

**Why it matters:** This is the fundamental temporal split that prevents data leakage. Each user's interaction history is sorted chronologically, then split 70/15/15. This mirrors the real world: you can only train on past data and evaluate on future interactions. Every user has at least 1 train event.

**What it does:**
1. Groups interactions by `user_id`
2. Sorts each user's events by `date`
3. Takes the earliest ~70% for training, next ~15% for validation, last ~15% for testing
4. Returns `(train_df, val_df, test_df)`

### Concepts in this stage

| # | Concept | What it is |
|---|---|---|
| 18 | **Per-user temporal split** (70/15/15) | Chronological per-user split. Every user contributes to all three splits. This is fundamentally different from a random split — it respects the arrow of time. |
| 19 | **Collaborative support filtering** (iterative) | Only items with ≥5 positive train interactions and users with ≥5 events & ≥2 positives survive. Iterated until stable (cross-filtering). Val/test filtered to match — they never influence what's retained. |
| 20 | **Leakage-safe prefix histories** | Training examples use only interactions *before* the target item. Future events (including the target itself) are excluded — no temporal leakage. |
| 21 | **Progressive history for eval** | Validation uses train history only. Test uses train + val history. Matches real deployment: more history accumulates over time. |
| 22 | **Release-aware candidate filtering** | Items released *after* a user's last train interaction date are excluded. Prevents recommending games from the future. |
| 23 | **Triple catalog system** | `fallback_games` ⊃ `full_games` ⊃ `warm_games`. Each serves a purpose: cold-start fallback, item tower input, retrieval candidates. |

---

## Stage 4: Collaborative Filtering

### What happens here

After temporal split, the catalog is pruned to "warm" items and users — those with sufficient interaction signal for collaborative learning. This is iterative cross-filtering: removing an item may cause a user to drop below threshold, and vice versa.

### Functions

#### `retain_train_supported_catalog(train_frame, min_item_positives, min_user_events, min_user_positives)`
**Cell:** 25

**Signature:**
```python
def retain_train_supported_catalog(
    train_frame: pd.DataFrame,
    min_item_positives: int = MIN_TRAIN_ITEM_POSITIVES,
    min_user_events: int = MIN_TRAIN_USER_EVENTS,
    min_user_positives: int = MIN_TRAIN_USER_POSITIVES
) -> pd.DataFrame
```

**Why it matters:** Not all users and items have enough signal to learn meaningful embeddings. This prunes the catalog until only "warm" users/items remain, typically shrinking from ~15K items → ~2.5K warm items and 3K → ~2.9K warm users.

**What it does:**
1. `while` loop: filter items with < `min_item_positives` positive interactions
2. Filter users with < `min_user_events` total events OR < `min_user_positives` positive events
3. Exit when DataFrame shape stabilizes (converged)
4. Returns the final warm training frame

### Concepts in this stage

| # | Concept | What it is |
|---|---|---|
| 19 | **Collaborative support filtering** | See Stage 3 above. This pruning step is essential — without it, the model wastes capacity on items/users with negligible signal. |

---

## Stage 5: Data Structures & Indexing

### What happens here

Interactions are transformed into the tensor-friendly data structures the model consumes: contiguous indices, history maps, seen-item sets, heldout positive maps, release-cutoff filters, and prefix training examples.

### Functions

#### `add_contiguous_indices(frame)`
**Cell:** 32

**Signature:**
```python
def add_contiguous_indices(frame: pd.DataFrame) -> pd.DataFrame
```

**Why it matters:** PyTorch `Embedding` layers expect integer indices from 0 to `num_embeddings - 1`. After filtering, IDs may have gaps. This remapping is essential for the model to work correctly without wasting embedding table space.

**What it does:** Adds `user_idx` and `item_idx` columns using `pd.factorize` (contiguous 0..N-1 mapping). Returns the DataFrame with both columns.

---

#### `positive_history_map(*frames)`
**Cell:** 34

**Signature:**
```python
def positive_history_map(*frames: pd.DataFrame) -> dict[int, list[int]]
```

**Why it matters:** The user tower builds each user's representation from items they've positively interacted with in the past (train only — val/test interactions are never seen during training).

**What it does:**
1. Filters to only positive interactions (`is_positive == 1`)
2. Groups by `user_idx`
3. Collects ordered lists of `item_idx` values (chronological from the sort in Stage 1)
4. Returns `dict[int, list[int]]` mapping `user_idx → [item_idx, ...]`

---

#### `seen_item_map(*frames)`
**Cell:** 34

**Signature:**
```python
def seen_item_map(*frames: pd.DataFrame) -> dict[int, set[int]]
```

**Why it matters:** At evaluation, we must mask out items the user has already interacted with — otherwise the model gets credit for "rediscovering" training items. This records every item each user has seen (positive or negative).

**What it does:** Collects the *set* of all `item_idx` values each user has interacted with across all provided frames. Returns `dict[int, set[int]]`.

---

#### `heldout_positive_map(frame, seen_items)`
**Cell:** 34

**Signature:**
```python
def heldout_positive_map(
    frame: pd.DataFrame,
    seen_items: dict[int, set[int]]
) -> dict[int, list[int]]
```

**Why it matters:** Evaluation metrics require knowing the ground-truth positives in val/test. This isolates only positive interactions, excluding items already seen in training.

**What it does:**
1. Filters `frame` to `is_positive == 1`
2. Removes items present in the user's `seen_items` set
3. Returns `dict[int, list[int]]`

---

#### `filter_targets_released_by_cutoff(positive_targets, candidate_cutoffs)`
**Cell:** 35

**Signature:**
```python
def filter_targets_released_by_cutoff(
    positive_targets: dict[int, list[int]],
    candidate_cutoffs: dict[int, pd.Timestamp]
) -> dict[int, list[int]]
```

**Why it matters:** A user cannot interact with a game before its release. Including unreleased-at-eval-time games as "missed recommendations" would unfairly penalize the model. This is a critical temporal constraint.

**What it does:** For each user, filters heldout positives to only items with `date_release ≤ user's last interaction date`. Returns filtered targets.

---

#### `build_prefix_examples(history_map, max_history_length)`
**Cell:** 36

**Signature:**
```python
def build_prefix_examples(
    history_map: dict[int, list[int]],
    max_history_length: int = MAX_HISTORY_LENGTH
) -> list[dict]
```

**Why it matters:** Prefix training teaches the model to make good recommendations given *any prefix* of the user's history — making it robust to users with few interactions. The `max_history_length = 50` cap prevents self-attention from becoming O(n²) expensive.

**What it does:**
1. For each user's positive history list, generates multiple training examples by sliding a window
2. At position `t` (starting from 1): prior items `[0:t]` are the history prefix, item `t` is the positive target
3. Items older than 50 positions are dropped
4. Returns `list[{'user_idx', 'history_item_idxs', 'positive_item_idx'}]`

---

#### `SteamSequenceTwoTowerDataset(Dataset)`
**Cell:** 37

**Class definition:**
```python
class SteamSequenceTwoTowerDataset(torch.utils.data.Dataset)
```

**Methods:**

| Method | Signature |
|---|---|
| `__init__` | `(self, prefix_examples, num_items, max_history_length, user_id_to_idx, item_id_to_idx, padding_item_idx)` |
| `__len__` | `(self) -> int` |
| `__getitem__` | `(self, index) -> tuple[Tensor, Tensor, Tensor]` |

**Why it matters:** Standard PyTorch `Dataset` wrapper. History sequences padded to `max_history_length = 50` in `__getitem__`, so the DataLoader's default `torch.stack` collation works without a custom `collate_fn`.

**What `__getitem__` does:**
1. Retrieves the prefix example at `index`
2. Truncates history to last 50 items
3. Right-pads with `padding_item_idx` to exactly 50 positions
4. Returns `(user_idx, padded_history, target_item_idx)` as PyTorch tensors

### Concepts in this stage

| # | Concept | What it is |
|---|---|---|
| 24 | **Contiguous 0-based indexing** | `user_id` and `app_id` → 0..N-1 for embedding table lookup. |
| 25 | **Padding index** | `padding_item_idx = num_items` (one beyond last real item). Right-pads histories to 50. |
| 26 | **`train_pos_mask`** | Boolean matrix `(num_users, num_items)` marking all train-positive pairs. Masks false negatives during training. Formerly `user_positive_matrix`. |
| 27 | **`positive_history_map`** | Dict `{user_idx: [item_idx, ...]}` — chronological positive items for prefix examples and eval histories. |
| 28 | **`seen_item_map`** | Dict `{user_idx: {item_idx, ...}}` — all items the user has interacted with. Masks already-seen items at evaluation. |
| 29 | **`heldout_positive_map`** | Positive targets for val/test that the user hasn't seen in training. Removes items already in `seen_items`. |
| 30 | **`val_candidate_cutoffs` / `test_candidate_cutoffs`** | Dict `{user_idx: max_date}` — latest interaction date in train (or train+val). Items released after this are not eligible candidates. |

### Connection: how data structures flow through the pipeline

```
train_interactions ──┬──→ positive_history_map ──→ build_prefix_examples ──→ Dataset ──→ DataLoader
                     │
                     ├──→ seen_item_map (fed to evaluation)
                     │
                     └──→ train_pos_mask (fed to train_epoch for false-negative masking)

val_interactions ──────→ heldout_positive_map ──→ filter_targets_by_release_date ──→ evaluate_filtered
                     │
                     └──→ seen_item_map (adds to train seen items)

test_interactions ─────→ same flow as val, but also accumulates train+val history
```

---

## Stage 6: Model Architecture — `TwoTowerModel`

### What happens here

The two-tower architecture is defined: a user tower that encodes interaction history via self-attention, an item tower that encodes static features + ID embedding, and a shared L2-normalized embedding space where dot product = similarity.

### Functions

#### `TwoTowerModel(nn.Module)`
**Cell:** 38

```python
class TwoTowerModel(nn.Module)
```

**Why it matters:** This is the core of the entire recommendation system. Separating user and item representations enables efficient ANN retrieval at serving time: item embeddings are precomputed offline, user embeddings are computed on-the-fly from interaction history.

**Architecture diagram:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TwoTowerModel                                │
│                                                                     │
│   USER TOWER                         ITEM TOWER                     │
│   ┌──────────────────┐              ┌──────────────────────┐       │
│   │ user_id_emb      │              │ item_id_emb (128-dim)│       │
│   │ (128-dim lookup) │              │                      │       │
│   └────────┬─────────┘              │ + static_features    │       │
│            │                        │   (1,173-dim)        │       │
│   ┌────────▼─────────┐              └──────────┬───────────┘       │
│   │ history items     │                         │                   │
│   │ → item_id_emb     │              ┌──────────▼───────────┐       │
│   │   (shared vocab)  │              │ Linear(128+1173→256) │       │
│   │ → MultiHeadAttn   │              │ LayerNorm            │       │
│   │   4 heads, 128-dim│              │ GELU                 │       │
│   │   + padding mask  │              │ Dropout(0.2)         │       │
│   │ → Residual+LN     │              │ Linear(256→128)      │       │
│   │ → Mean pool       │              │ L2-normalize         │       │
│   │   (excl. padding) │              └──────────┬───────────┘       │
│   └────────┬─────────┘                          │                   │
│            │                                    │                   │
│   sequence_vector (128-dim)            item_vector (128-dim)        │
│            │                                    │                   │
│   ┌────────▼─────────┐                          │                   │
│   │ concat(user_id,   │                          │                   │
│   │   sequence_vector)│                          │                   │
│   │ Linear(256→256)   │                          │                   │
│   │ LayerNorm         │                          │                   │
│   │ GELU              │                          │                   │
│   │ Dropout(0.2)      │                          │                   │
│   │ Linear(256→128)   │                          │                   │
│   │ L2-normalize      │                          │                   │
│   └────────┬─────────┘                          │                   │
│            │                                    │                   │
│   user_vector (128-dim)                         │                   │
│            │                                    │                   │
│            └──────────────┬─────────────────────┘                   │
│                           │                                         │
│                  ┌────────▼────────┐                                │
│                  │ dot_product     │                                │
│                  │ + item_bias     │                                │
│                  │ → logit         │                                │
│                  └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

**Methods (all in cell 38):**

##### `__init__(num_users, num_items, item_feature_dim, embedding_dim, num_heads, dropout)`

Initializes:
- `user_id_embedding`: `nn.Embedding(num_users, 128)`
- `item_id_embedding`: `nn.Embedding(num_items, 128)` — shared between user history and item tower
- Self-attention: `nn.MultiheadAttention(128, num_heads=4, dropout=0.2, batch_first=True)`
- User tower projections: `Linear(256→256) → LayerNorm → GELU → Dropout → Linear(256→128)`
- Item tower projections: `Linear(128+item_feature_dim→256) → LayerNorm → GELU → Dropout → Linear(256→128)`
- Item bias: `nn.Embedding(num_items, 1)` initialized to zero
- Padding item index: `num_items` (one past last real item)

##### `encode_item(item_ids, item_features) -> Tensor`
```python
def encode_item(self, item_ids: Tensor, item_features: Tensor) -> Tensor  # [B, 128]
```
1. Looks up `item_id_embedding[item_ids]`
2. Concatenates with `item_features`
3. Projects through `Linear → LayerNorm → GELU → Dropout → Linear`
4. L2-normalizes output
5. Returns `[batch_size, 128]`

##### `encode_user(user_ids, history_item_ids) -> Tensor`
```python
def encode_user(
    self,
    user_ids: Tensor,                    # [B]
    history_item_ids: Tensor,            # [B, 50]
) -> Tensor                              # [B, 128]
```
1. Looks up `item_id_embedding[history_item_ids]` using the shared vocabulary
2. Applies multi-head self-attention with `key_padding_mask` (derived from padding_idx)
3. Residual connection + LayerNorm
4. Mean-pools over non-padding positions → sequence vector
5. User ID embedding (or zeros in no_user_id ablation)
6. Concatenates user ID embedding + sequence vector
7. Projects through user tower `Linear → LN → GELU → Dropout → Linear`
8. L2-normalizes output
9. Returns `[batch_size, 128]`

##### `encode_all_items(item_features) -> Tensor`
```python
def encode_all_items(self, item_features: Tensor) -> Tensor  # [num_items, 128]
```
Precomputes embeddings for all items in the warm catalog. Caller does not need to pass
item IDs — the method constructs them internally and passes them to `encode_item`.
Used at evaluation for full-catalog scoring.

### Concepts in this stage

| # | Concept | What it is |
|---|---|---|
| 31 | **Two-tower architecture** | Separate user/item encoders → shared L2-normalized space. Retrieval = nearest-neighbor search. |
| 32 | **Item tower** | `concat(item_id_emb, static_features) → Linear(128+1173→256) → LN → GELU → Dropout(0.2) → Linear(256→128) → L2-norm` |
| 33 | **User ID embedding** | Learnable 128-dim embedding per user (standard collaborative filtering lookup). |
| 34 | **Shared item-ID vocabulary** | `item_id_embedding` used for BOTH item tower (candidate encoding) AND history sequence encoding. Keeps representations consistent. |
| 35 | **Multi-head self-attention over history** | `MultiheadAttention(128, heads=4, dropout=0.2, batch_first=True)`. Each history item attends to every other — captures pairwise relationships (e.g., "this user likes indie RPGs"). |
| 36 | **Padding mask in attention** | `key_padding_mask=True` for padding positions → zero attention weight. |
| 37 | **Residual connection + LayerNorm** | `attended = LN(history_vectors + attention_output)` for training stability. |
| 38 | **Mean pooling over valid positions** | Sequence vector = mean of attention outputs over non-padding positions. Zero-history users get a zero vector. |
| 39 | **User tower** | `concat(user_id_emb, sequence_vec) → Linear(256→256) → LN → GELU → Dropout(0.2) → Linear(256→128) → L2-norm` |
| 40 | **Item bias** | `nn.Embedding(num_items, 1)` initialized to zero. Learns broad appeal per item. Added to logits at both train and inference. User bias intentionally omitted (cancels row-wise in softmax). |
| 41 | **`encode_all_items`** | Precomputes every item embedding in one pass. Used for evaluation-time full-catalog scoring. |

### Connection: how the towers interact

1. **During training:** Both towers are called per batch. User vectors and positive item vectors are returned. Dot product yields logits. The shared `item_id_embedding` means history items and candidate items use the same embedding space — the attention mechanism can cross-reference items the user liked with items being scored.

2. **During evaluation:** `encode_all_items` precomputes all item embeddings once (no item IDs needed — just static features). Then each user is encoded via `encode_user` using their history, and scored against all precomputed item vectors via a single matrix multiplication.

3. **The item bias** is a scalar per item that learns broad appeal. An item everyone likes gets a positive bias, independent of personalization. This is added at both training and inference.

---

## Stage 7: Training

### What happens here

The training loop implements masked in-batch InfoNCE with false-negative masking, LogQ popularity correction, temperature scaling, gradient clipping, learning rate scheduling, and early stopping.

### Functions

#### `train_epoch(model, dataloader, optimizer, item_features, positive_matrix, device, log_q)`
**Cell:** 39

**Signature:**
```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    item_features: Tensor,            # [num_items, item_feature_dim]
    positive_matrix: Tensor,          # [num_users, num_items] boolean
    device: torch.device,
    log_q: Tensor | None = None,      # [num_items] — LogQ correction values
) -> float                            # average loss
```

**Why it matters:** Implements masked in-batch InfoNCE with false-negative masking and LogQ correction for popularity debiasing. Temperature is a learnable model parameter (`model.temperature()`), not a fixed hyperparameter.

**Training algorithm (step by step):**

For each batch `(user_ids, history_item_ids, pos_item_ids)`:

1. **Encode users:** `user_vecs = model.encode_user(user_ids, history_item_ids)` → `[B, 128]`
2. **Encode positive items:** `item_vecs = model.encode_item(pos_item_ids, item_features[pos_item_ids])` → `[B, 128]`
3. **Compute logits:** `logits = user_vecs @ item_vecs.T` → `[B, B]`
   - Diagonal: positive pairs (user i matched with item i)
   - Off-diagonal: in-batch negatives
4. **Add item bias:** `logits += model.item_bias(pos_item_ids).squeeze(-1).unsqueeze(0)`
5. **Mask false negatives:**
   - Build mask: for each (user_i, item_j) pair, if `positive_matrix[user_i, item_j]` is True AND i ≠ j → mask to `-1e9`
   - This prevents items the user actually likes (in train) from being treated as negatives
6. **Apply LogQ correction:** If `log_q` is provided: `logits -= log_q[pos_item_ids]`
   - Corrects for popular items being over-sampled as in-batch negatives
7. **Cross-entropy loss:** `labels = [0, 1, 2, ..., B-1]` (diagonal targets)
8. **Backward pass + gradient clipping:** `clip_grad_norm_(max_norm=1.0)`
9. Returns average loss

### Concepts in this stage

| # | Concept | What it is |
|---|---|---|
| 42 | **In-batch InfoNCE (contrastive loss)** | `user_vecs @ item_vecs.T` → logit matrix. Diagonal = positives, off-diagonal = negatives. CE with diagonal targets. |
| 43 | **False negative masking** | Train-positive (user, item) pairs appearing off-diagonal are masked to `-1e9` — prevents contradictory training signal. |
| 44 | **Learnable temperature** | Logits divided by `model.temperature()` before softmax. The temperature is a learnable parameter (`log_temperature`) initialized at `log(0.07)`, allowing the model to tune its own sharpness during training. Lower = sharper distribution, harder negatives. |
| 45 | **LogQ correction (in-batch sampling debiasing)** | `log_q[item] = log(1 - (1 - p_item)^B)` where `p_item = count(item)/total` and B = batch size. Accounts for an item appearing as a negative in up to B−1 other rows, so its true sampling probability is higher than marginal `p_j`. Subtracted from logits before CE. (See Yi et al. 2019, "Sampling-Bias-Corrected Neural Modeling"). |
| 46 | **Singleton batch skip** | Batches with < 2 examples are skipped (no in-batch contrast possible). |
| 47 | **Gradient clipping** | `clip_grad_norm_(max_norm=1.0)` — prevents exploding gradients. |
| 48 | **AdamW optimizer** | `lr=3e-4, weight_decay=1e-4` — decoupled weight decay for better generalization. |
| 71 | **Experiment tracking** | Per-run header with config summary, per-epoch timing + `*` best-epoch marker, and final summary (best epoch, recall, total time, final LR). |
| 72 | **Training curves plot** | Cell 44: dual-axis matplotlib plot — training loss (blue, left y-axis) + validation Recall@10 (red, right y-axis) over epochs, with best-epoch marker and annotation. |
| 49 | **ReduceLROnPlateau scheduler** | `mode="max"`, reduces LR when validation Recall@10 plateaus for 3 epochs. Monitors Recall@10 (not loss) because InfoNCE loss is batch-dependent and doesn't measure full-catalog retrieval quality — the scheduler must align with the actual objective. |
| 50 | **Early stopping** | Stops after 6 epochs without validation Recall@10 improvement. |
| 51 | **Best checkpoint selection** | Model saved at epoch with highest validation Recall@10 (not loss). |

### Connection: training data flow

```
DataLoader
    │
    ├──→ user_ids ──→ user_id_embedding ──┐
    ├──→ history_item_ids ──→ item_id_emb │──→ MultiHeadAttn ──→ seq_vec ──→ concat ──→ Linear ──→ user_vec [B,128]
    │   (padded to 50)                    │
    │                                      │
    └──→ pos_item_ids ──→ item_id_emb ────┴──→ concat(static_features) ──→ Linear ──→ item_vec [B,128]

user_vec @ item_vec.T → logits [B,B]
logits = (logits / temperature) + item_bias
logits[masked_false_negatives] = -1e9
logits -= log_q[pos_item_ids]   ← corrects for popularity bias
loss = CrossEntropy(logits, [0,1,2,...,B-1])
```

---

## Stage 8: Evaluation

### What happens here

The trained model is evaluated against the full warm item catalog for each user. Per-user candidate filtering excludes already-seen items and not-yet-released items. Metrics: Recall@K, NDCG@K, HitRate@K with hybrid scoring (model + popularity blend).

The side-by-side comparison table in cell 43 runs all four methods for both val and test after training completes — the formatted table is the only results display, no separate cells needed.

### Functions

#### `padded_history_batch(user_ids, history_map, padding_idx, max_history_length)`
**Cell:** 40

**Signature:**
```python
def padded_history_batch(
    user_ids: list[int],
    history_map: dict[int, list[int]],
    padding_idx: int,
    max_history_length: int = MAX_HISTORY_LENGTH
) -> tuple[Tensor, Tensor]
```

**What it does:** Takes a batch of user indices, looks up their positive history, right-pads to `max_history_length` with `padding_idx`, returns `(padded_history, lengths)` as tensors.

---

#### `available_candidate_indices(user_idx, seen_items, candidate_cutoffs, item_release_dates)`
**Cell:** 40

**Signature:**
```python
def available_candidate_indices(
    user_idx: int,
    seen_items: dict[int, set[int]],
    candidate_cutoffs: dict[int, np.datetime64 | np.ndarray],
    item_release_dates: np.ndarray,
) -> np.ndarray  # [num_available] int64 indices
```

**What it does:** Returns an array of item indices available for this user:
- Excludes items in `seen_items[user_idx]` (already interacted)
- Excludes items released after `candidate_cutoffs[user_idx]` (released in the future relative to eval point)

---

#### `evaluate_filtered(model, positive_targets, user_histories, seen_items, candidate_cutoffs, item_features, item_release_dates, device, ...)`
**Cell:** 41

**Usage:** Called 4 times in evaluation stage (val + test for comparison table) and once per epoch for validation recall tracking.

**Signature:**
```python
def evaluate_filtered(
    model: torch.nn.Module,
    positive_targets: dict[int, list[int]],
    user_histories: dict[int, list[int]],
    seen_items: dict[int, set[int]],
    candidate_cutoffs: dict[int, np.datetime64 | np.ndarray],
    item_features: torch.Tensor,
    item_release_dates: np.ndarray,
    device: torch.device,
    k: int = EVAL_K,
    user_batch_size: int = EVAL_USER_BATCH_SIZE,
    hybrid_alpha: float = HYBRID_ALPHA,
    item_popularity_log: torch.Tensor | None = None,
    padding_idx: int | None = None,
) -> dict[str, float]
```

**Why it matters:** This is the primary evaluation function. Scores every user against the entire warm catalog, applies per-user availability masks, and computes Recall@K, Precision@K, and HitRate@K. Evaluation is release-aware (unreleased items excluded) and seen-item-filtered — matching real-world serving.

**Algorithm:**
1. Precompute all item embeddings via `model.encode_all_items(item_features)`
2. For each user (batched):
   a. Build padded history from `user_histories`
   b. Encode user: `user_vec = model.encode_user(user_ids, history_item_ids)`
   c. Score all items: `scores = user_vec @ all_item_vecs.T + item_bias`
   d. Apply hybrid scoring: `scores += HYBRID_ALPHA × item_popularity_log`
   e. Apply availability mask (seen items + release cutoff → filtered)
   f. Top-K retrieval
   g. Compare against `positive_targets[user_idx]`
3. Aggregate Recall@K, Precision@K, HitRate@K across all users
4. Returns metrics dict

### Concepts in this stage

| # | Concept | What it is |
|---|---|---|
| 52 | **Full-catalog warm retrieval** | Each user scored against ALL warm catalog items (not just a batch). Realistic deployment scenario. |
| 53 | **Release-aware candidate masking** | Items released after user's cutoff date excluded from candidate pool. |
| 54 | **Seen-item masking** | Items the user already interacted with excluded from recommendations. |
| 55 | **Metrics: HitRate@10, Precision@10, Recall@10** | Standard retrieval metrics at k=10. Recall is primary (best reflects how many relevant items are surfaced). |
| 56 | **Hybrid scoring** | `scores = model_scores + item_bias + α × log(1 + train_popularity)`. `HYBRID_ALPHA = 0.2` (tuned via sweep on no_user_id model — optimal blend of popularity with learned personalization). |

### Connection: evaluation uses two different history windows

- **Validation:** users are evaluated using only train history → `history_map` = train positives only
- **Test:** users are evaluated using train + val history → `history_map` = train + val positives (progressive history matches real deployment where more data accumulates)

---

## Stage 9: Baselines

### What happens here

Three baselines provide context for the two-tower model's performance: a simple popularity baseline (upper bound on non-personalized), a random-expected baseline (statistical floor), and a content-centroid baseline (tests whether static features alone suffice).

### Functions

#### `evaluate_popularity_baseline(positive_targets, user_histories, seen_items, candidate_cutoffs, popular_items, item_release_dates, k)`
**Cell:** 45

**Signature:**
```python
def evaluate_popularity_baseline(
    positive_targets: dict[int, list[int]],
    user_histories: dict[int, list[int]],
    seen_items: dict[int, set[int]],
    candidate_cutoffs: dict,
    popular_items: list[int],
    item_release_dates: np.ndarray | None = None,
    k: int = EVAL_K,
) -> dict[str, float]
```

**Why it matters:** The simplest possible baseline — if the model can't beat "recommend the most popular items in train," it hasn't learned anything useful.

**What it does:** For each user, recommends the top-K items by train interaction frequency (excluding already-seen and unreleased items). Computes HitRate@K, Precision@K, and Recall@K against heldout positives.

---

#### `evaluate_random_expected_baseline(positive_targets, user_histories, seen_items, candidate_cutoffs, item_release_dates, k)`
**Cell:** 46

**Signature:**
```python
def evaluate_random_expected_baseline(
    positive_targets: dict[int, list[int]],
    user_histories: dict[int, list[int]],
    seen_items: dict[int, set[int]],
    candidate_cutoffs: dict,
    item_release_dates: np.ndarray | None = None,
    k: int = EVAL_K,
) -> dict[str, float]
```

**Why it matters:** Provides the theoretical floor for recommendation quality. Uses the hypergeometric distribution to analytically compute expected hits from random selection.

**What it does:** `ExpectedHits = K × n / M` where K = recommendation cutoff, n = number of true positives, M = number of available candidates. Returns expected HitRate@K, Precision@K, Recall@K.

---

#### `evaluate_content_centroid_baseline(positive_targets, user_histories, seen_items, candidate_cutoffs, item_features, item_release_dates, k)`
**Cell:** 47

**Signature:**
```python
def evaluate_content_centroid_baseline(
    positive_targets: dict[int, list[int]],
    user_histories: dict[int, list[int]],
    seen_items: dict[int, set[int]],
    candidate_cutoffs: dict,
    item_features: torch.Tensor,
    item_release_dates: np.ndarray | None = None,
    k: int = EVAL_K,
) -> dict[str, float]
```

**Why it matters:** A non-trivial content-based baseline. Tests whether simple cosine similarity to a user's average item features can perform well without collaborative learning. If this baseline is strong, static features are highly informative on their own.

**What it does:**
1. For each user, computes the mean of their training item feature vectors (the "centroid")
2. Scores all candidates by cosine similarity to this centroid
3. Excludes already-seen and unreleased items
4. Computes HitRate@K, Precision@K, Recall@K

### Concepts in this stage

| # | Concept | What it is |
|---|---|---|
| 57 | **Popularity baseline** | Top-K most-interacted items per user. Upper bound on non-personalized performance. |
| 58 | **Random expected baseline** | Analytical expected value of random recommendation (hypergeometric). Lower bound. |
| 59 | **Content centroid baseline** | Averages user's history item features, recommends closest items by cosine similarity. Tests static content sufficiency. |

---

## Stage 10: Cold Start

### What happens here

New items with no interaction history cannot be served by collaborative filtering (no learned embedding). The cold-start fallback uses popularity and quality signals from game metadata to make these items discoverable.

### Functions

#### `recommend_cold_start_v1(games_data, train_interactions, n)`
**Cell:** 50

**Signature:**
```python
def recommend_cold_start_v1(
    games_data: pd.DataFrame,
    train_interactions: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame
```

**Why it matters:** New items with no interaction history cannot be served by collaborative filtering (no embedding). This fallback ensures these items are still discoverable using popularity and quality signals from metadata alone, maintaining catalog coverage.

**What it does:**
1. Computes item popularity from `train_interactions` (pre-warm-filter train data)
2. Filters to items in `games_data` (the `fallback_games` catalog)
3. Computes weighted score: `0.60 × popularity_rank + 0.25 × positive_ratio_rank + 0.15 × review_count_rank`
4. Returns top-N recommendations

### Concepts in this stage

| # | Concept | What it is |
|---|---|---|
| 4 | **Cold-start fallback catalog** | `fallback_games` preserves the broader valid catalog (pre-warm-filter) for recommending items that lack collaborative signal. |
| 60 | **Cold-start recommendation** | Weighted score combining popularity, positive ratio, and review count ranks. Uses fallback catalog + pre-warm-filter train data. |

---

## Stage 11: Learning Dynamics & Geometry

These are conceptual — no dedicated functions, but they explain *why* the training works.

| # | Concept | What it is |
|---|---|---|
| 61 | **Contrastive learning as geometry sculpting** | InfoNCE shapes the embedding space so positive pairs are pulled together and negatives pushed apart. Creates a metric space where ANN retrieval is meaningful. |
| 62 | **Relative preference ordering** | The model doesn't predict ratings — it learns that "for this user, item A should rank above items B, C, D...". The ranking is relative, not absolute. |
| 63 | **Popularity bias in in-batch negatives** | Popular items appear more often as positives → appear more often as in-batch negatives → model is pushed to down-rank popular items. LogQ correction counteracts this. |
| 64 | **Item bias as broad appeal** | Learnable scalar per item captures general appeal independent of user. Added to scores at both train and inference. |

---

## Stage 12: Reproducibility & Infrastructure

| # | Concept | What it is |
|---|---|---|
| 65 | **Fixed random seeds** | `RANDOM_SEED = 42` set for `random`, `numpy`, `torch`, `torch.cuda`, DataLoader generator, and reservoir sampling. |
| 66 | **Run manifest** | JSON with seed, user/item counts, filtering thresholds — exported alongside processed data for reproducibility. |
| 67 | **Artifact export** | `full_games`, `warm_games`, `fallback_games`, `item_features_df`, and split CSVs saved to `v1/data/processed/`. Model checkpoint + FAISS index saved to `v1/models/`. Both directories are produced by running the notebook (`os.makedirs` at cells 0, 30, 43) — they are not tracked in version control. |

---

## Stage 13: Experimental Progression

Design evolution from earlier versions.

| # | Concept | What it is |
|---|---|---|
| 68 | **GRU → Attention replacement** | Original user tower used GRU over history. Replaced with multi-head self-attention — captures pairwise item relationships across full history. |
| 69 | **GRU downsides** | Processes items sequentially, losing long-range signal. Struggles with variable-length sequences and distant dependencies. |
| 70 | **Attention benefits** | O(L²) pairwise attention over all history items simultaneously. Can learn "this user likes indie RPGs" by cross-referencing genre+tags across items. |

---

## Stage 14: HYBRID_ALPHA Tuning

### What happens here

After the best model architecture is determined (no_user_id), HYBRID_ALPHA is swept over {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
on the trained model (no retraining needed — HYBRID_ALPHA only affects evaluation scoring).
The sweep found α=0.2 gives best test Recall@10 (0.0856), a +2.3% gain over the previous default of 0.3.

**Cell:** 48

| α | Val R@10 | Test R@10 |
|---|---|---|
| 0.0 | 0.0923 | 0.0784 |
| 0.1 | 0.0926 | 0.0843 |
| **0.2** | **0.0934** | **0.0856** ← best |
| 0.3 | 0.0910 | 0.0837 |
| 0.4 | 0.0890 | 0.0840 |
| 0.5 | 0.0889 | 0.0821 |

### Concepts in this stage

| # | Concept | What it is |
|---|---|---|
| 73 | **HYBRID_ALPHA sweep** | Parameter sweep over α ∈ [0, 0.5] on a trained model. No retraining needed — HYBRID_ALPHA only affects the evaluation scoring function. Cell 48 runs the sweep and prints a formatted comparison table. |

---

## Stage 15: Ablation Experiments

### What happens here

Three controlled experiments isolate the contribution of each major model component:
hybrid popularity scoring, per-user collaborative embedding, and multi-head self-attention.
The `ABLATION` variable in cell 0 controls which variant runs.

**Cells:** 0 (config), 1 (cleanup), 39 (model), 43 (setup), 44 (HYBRID_ALPHA), 47 (training), 49 (summary)

### Ablation modes

| ABLATION | What changes | Question answered |
|---|---|---|
| `"full"` | Default v1 — all components active | Baseline |
| `"no_popularity"` | `HYBRID_ALPHA = 0` — no popularity blend at inference | How much recall comes from personalization vs heuristic? |
| `"no_user_id"` | Remove `user_id_embedding` from user tower | Does the model generalize or memorize warm-user identity? |
| `"mean_pool"` | Replace `MultiHeadAttention` with mean pooling | Does self-attention earn its keep at this scale? |

### Concepts in this stage

| # | Concept | What it is |
|---|---|---|
| 74 | **Ablation framework** | Single `ABLATION` variable controls model variant + inference config. `ablation_results.json` persists metrics across runs for cumulative comparison. |
| 75 | **Cleanup between runs** | Cell 1 provides a commented-out `shutil.rmtree` block to clear `processed/` and `models/` before re-running with a different ablation. |
| 76 | **Ablation summary table** | Cell 50 reads `ablation_results.json` and prints a cumulative side-by-side table of all completed ablation runs with full metric breakdowns. |

---

## Detailed Connection Walkthrough

Here is how every term you identified connects through the pipeline:

### 1. Temporal Split → Collaborative Filtering → Dataset

```
Raw interactions (all users, all items)
    │
    ▼
temporal_split_by_user()  ← per-user chronological 70/15/15 cut
    │
    ├──→ train_interactions
    ├──→ val_interactions
    └──→ test_interactions
            │
            ▼
fit_engagement_transform(train)  ← clip@99% + MinMaxScaler on train only
add_engagement_strength(all)     ← apply train-fit transforms to val/test (no leakage)
            │
            ▼
retain_train_supported_catalog(train)  ← iterative cross-filter
    │                                       items ≥5 positives, users ≥5 events + ≥2 positives
    │                                       converges when stable
    ▼
warm catalog (∼2.5K items, ∼2.9K users)
    │
    ▼
add_contiguous_indices(all splits)  ← factorize user_id/app_id → 0..N-1
    │
    ├──→ positive_history_map(train)  ← {user_idx: [item_idx, ...]}  chronologically ordered
    ├──→ seen_item_map(train)         ← {user_idx: {item_idx, ...}}  all interacted items
    ├──→ heldout_positive_map(val, seen_items)  ← positives in val NOT already seen
    ├──→ filter_targets_released_by_cutoff(heldout, cutoffs)  ← remove future items
    └──→ build_prefix_examples(history_map)  ← sliding window over each user's history
```

### 2. Prefix Examples → Dataset → DataLoader

```
build_prefix_examples
    │   For user with history [A, B, C, D]:
    │   prefix[1]: history=[A],        target=B
    │   prefix[2]: history=[A,B],      target=C
    │   prefix[3]: history=[A,B,C],    target=D
    │   (items beyond 50 dropped)
    │
    ▼
list of {user_idx, history_item_idxs, positive_item_idx}
    │
    ▼
SteamSequenceTwoTowerDataset
    │   __getitem__ truncates to last 50, right-pads with padding_item_idx
    │   returns (user_idx, padded_history[50], target_item_idx)
    │
    ▼
DataLoader(batch_size=512, shuffle=True)
    │   torch.stack collates into [B], [B,50], [B]
    ▼
```

### 3. Two-Tower Model: itemid → item tower, history attention → user tower

```
BATCH: (user_ids [B], history_item_ids [B,50], target_item_ids [B])

┌─────────────────────────────────────────────────────────────┐
│ ITEM TOWER (called for target items AND history items)      │
│                                                             │
│   item_id_embedding[item_ids]         → [*, 128]           │
│   static_features[item_ids]           → [*, 1173]          │
│   concat → Linear(128+1173→256)       → [*, 256]           │
│   LayerNorm → GELU → Dropout(0.2)     → [*, 256]           │
│   Linear(256→128)                     → [*, 128]           │
│   L2-normalize                        → [*, 128]           │
│                                                             │
│   For history items: called once → history_vectors [B,50,128]│
│   For target items:  called once → positive_item_vecs [B,128]│
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ USER TOWER                                                  │
│                                                             │
│   user_id_embedding[user_ids]         → [B, 128]           │
│                                                             │
│   history_vectors [B, 50, 128]                              │
│     → MultiHeadAttention(heads=4, dropout=0.2)              │
│        key_padding_mask = history_mask (True for padding)   │
│     → attended [B, 50, 128]                                │
│     → Residual: history_vectors + attended                 │
│     → LayerNorm                                             │
│     → Mean pool over valid (non-padding) positions          │
│     → sequence_vector [B, 128]                             │
│                                                             │
│   concat(user_id_emb, sequence_vector) → [B, 256]          │
│   Linear(256→256) → LN → GELU → Dropout → Linear(256→128)  │
│   L2-normalize                        → user_vec [B, 128]  │
└─────────────────────────────────────────────────────────────┘
```

### 4. Train function: InfoNCE + false negative mask + LogQ + temperature

```
user_vec [B, 128]  @  positive_item_vec [B, 128].T  →  logits [B, B]

For each cell logits[i][j]:
  - i == j  → positive pair (user i matched with their target item)
  - i != j  → in-batch negative (item j is not user i's target)

logits = logits / temperature            ← sharpen distribution (0.15)
logits += item_bias[target_items]        ← add broad-appeal correction

false_negative_mask: for (i,j) where train_pos_mask[user_i, item_j] is True AND i != j:
  logits[i][j] = -1e9                   ← don't penalize true positives

logits -= log_q[target_items]           ← correct for popularity bias

labels = [0, 1, 2, ..., B-1]            ← diagonal targets
loss = CrossEntropyLoss(logits, labels)  ← InfoNCE
```

**Why false negative masking matters:** If user 3's training history includes item 7, and item 7 appears as an in-batch negative in a batch where user 3 is present, without masking the model would be told "user 3 should NOT like item 7" — contradicting the training data. The mask prevents this.

**Why LogQ correction matters:** Popular items appear in more batches as positive targets → appear more often as in-batch negatives → the model learns to push them down for ALL users. LogQ correction subtracts `log(frequency)` from the logits, compensating for this sampling bias.

### 5. Evaluation: full-catalog scoring + per-user masking

```
encode_all_items(item_features_matrix) → all_item_vecs [num_items, 128]
all_item_biases = item_bias(all_item_ids).squeeze()  [num_items]

For each user (in batches):
  user_vec = encode_user(user_idx, history, history_mask)  [1, 128]
  
  scores = user_vec @ all_item_vecs.T + all_item_biases         [num_items]
  scores += HYBRID_ALPHA * log(1 + train_popularity)            hybrid blending
  
  mask = available_candidate_indices(user_idx, seen_map, cutoffs) [num_items bool]
    - False: items in seen_map[user_idx]        (already interacted)
    - False: items released after cutoffs[user_idx] (future leak)
  
  scores[~mask] = -inf
  
  top_k = scores.topk(K).indices
  
  recall@K = |top_k ∩ positive_targets[user_idx]| / |positive_targets[user_idx]|
```

### 6. Training loop: orchestration

```
MAX_EPOCHS = 30
best_val_recall = -float("inf")
patience_counter = 0
train_losses = []                        ← tracked for training curves plot
val_recalls = []                         ← tracked for training curves plot

Print experiment header (config summary + ablation label)

For epoch in 1..MAX_EPOCHS:
  ┌─ TRAIN ─────────────────────────────────────┐
  │  log_q = compute LogQ from train item counts │
  │  loss = train_epoch(model, train_loader,     │
  │      optimizer, item_features,               │
  │      train_pos_mask, device,                 │
  │      log_q=log_q)                            │
  └──────────────────────────────────────────────┘
  
  ┌─ VALIDATE ──────────────────────────────────┐
  │  metrics = evaluate_filtered(model, val_data,│
  │      device, item_features, history_map,     │
  │      seen_map, positive_targets, cutoffs)    │
  │  val_recall = metrics['Recall@10']           │
  └──────────────────────────────────────────────┘
  
  train_losses.append(loss)                ← track for plot
  val_recalls.append(val_recall)           ← track for plot

  scheduler.step(val_recall)               ← monitors Recall@10 (mode="max")
  
  is_best = "*" if val_recall > best_val_recall else " "
  if val_recall > best_val_recall:
    best_val_recall = val_recall
    patience_counter = 0
    save_checkpoint(model.state_dict())    ← best model by Recall@10
  else:
    patience_counter += 1
  
  Print per-epoch line with timing + is_best marker
  
  if patience_counter >= EARLY_STOPPING_PATIENCE (6):
    break

Restore best checkpoint
Print training summary (best epoch, recall, total time, final LR)
  │
  ▼
FAISS INDEX + COMPARISON TABLE (all in cell 43):
  Build FAISS index from item embeddings
  evaluate_filtered(val)                     ← two-tower val
  evaluate_filtered(test)                    ← two-tower test
  evaluate_popularity_baseline (val + test)  ← non-personalized upper bound
  evaluate_random_expected (val + test)      ← statistical floor
  evaluate_content_centroid (val + test)     ← static-features-only test
  Print side-by-side comparison table (model + 3 baselines for val + test)
  │
  ▼
TRAINING CURVES PLOT (cell 44):
  Dual-axis matplotlib: loss (blue) + val Recall@10 (red) over epochs
  │
  ▼
COLD START (cell 50):
  recommend_cold_start_v1(fallback_games)  ← items without collaborative signal
```

---

## Glossary

### Core terms

| Term | Definition |
|---|---|
| **Two-tower model** | Architecture with separate user and item encoders projecting into a shared L2-normalized embedding space. Retrieval = nearest-neighbor search. |
| **InfoNCE** | Information Noise-Contrastive Estimation. A contrastive loss that maximizes mutual information between positive pairs while minimizing it for negatives. |
| **In-batch negatives** | Other items in the same training batch, treated as negative examples. Scales with batch size — larger batches = more negatives = better training signal. |
| **False negative** | An item the user actually likes that appears as an in-batch negative. Must be masked to avoid contradictory training signal. |
| **LogQ correction** | Sampled softmax debiasing. Subtracts `log(P(item))` from logits to correct for popular items being over-represented as in-batch negatives. |
| **Temperature** | Scaling factor applied to logits before softmax. A learnable parameter initialized at `log(0.07)`, allowing the model to discover its optimal sharpness during training. |
| **Hybrid scoring** | Blending model scores with popularity signal at inference: `model_score + α × log(1 + popularity)`. Improves recall. |
| **L2-normalization** | Scaling vectors to unit length. Makes dot product equal to cosine similarity. Essential for the two-tower's metric-space retrieval. |
| **Self-attention** | Each element in a sequence attends to every other element, producing context-aware representations. Replaced GRU in the user tower. |
| **Padding mask** | Boolean mask flagging padded positions so attention ignores them. Prevents padding tokens from influencing the representation. |
| **Reservoir sampling** | Algorithm for uniform random sampling from a stream of unknown size. Each item has equal selection probability. |
| **Collaborative filtering** | Recommending items based on interaction patterns across users, rather than item content alone. |
| **Warm catalog** | Items with sufficient interaction signal for collaborative learning (≥5 positive interactions in train). |
| **Cold start** | Recommending items with no interaction history using only metadata signals (popularity, rating, reviews). |
| **Fallback catalog** | Broad catalog snapshot taken before warm filtering, used for cold-start recommendations. |
| **Temporal split** | Splitting interactions chronologically per user (not randomly). Prevents future-data leakage. |
| **Prefix training** | Generating multiple training examples per user by using varying amounts of history. Makes the model robust to users with few interactions. |
| **Item bias** | Learnable scalar per item capturing broad appeal. An item everyone likes gets a positive bias regardless of who's asking. |
| **User ID embedding** | Learnable vector per user capturing collaborative patterns. Combined with attention-based history encoding in the user tower. |
| **Shared vocabulary** | The same `item_id_embedding` table is used for both the item tower and the user's history encoding, keeping representations consistent. |

### Feature dimensions

| Component | Dimension | Description |
|---|---|---|
| Raw item features | 1,173 | Numeric (18, incl. 4 converted booleans) + tags (384) + descriptions (768) + flags (3) |
| Item ID embedding | 128 | Learned per-item collaborative embedding |
| User ID embedding | 128 | Learned per-user collaborative embedding |
| Shared embedding space | 128 | L2-normalized output of both towers |
| MiniLM tag embedding | 384 | Pre-trained SentenceTransformer |
| MPNet description embedding | 768 | Pre-trained SentenceTransformer |
| Self-attention heads | 4 | Multi-head attention over history |
| History length | 50 | Maximum history items (right-padded) |

### Key hyperparameters

| Parameter | Value | Stage |
|---|---|---|
| `TARGET_SAMPLED_ACTIVE_USERS` | 3,000 | Data sampling |
| `MIN_RAW_REVIEWS` | 20 | User filtering |
| `TRAIN_RATIO` / `VAL_RATIO` | 0.70 / 0.15 | Temporal split |
| `MIN_TRAIN_ITEM_POSITIVES` | 5 | Collaborative filtering |
| `MIN_TRAIN_USER_EVENTS` | 5 | Collaborative filtering |
| `MIN_TRAIN_USER_POSITIVES` | 2 | Collaborative filtering |
| `MAX_HISTORY_LENGTH` | 50 | Sequence modeling |
| `BATCH_SIZE` | 512 | Training |
| `EMBEDDING_DIM` | 128 | Model |
| `TEMPERATURE` | Learnable (init log(0.07)) | Training |
| `HYBRID_ALPHA` | 0.2 | Evaluation |
| `LEARNING_RATE` | 3e-4 | Training |
| `WEIGHT_DECAY` | 1e-4 | Training |
| `EARLY_STOPPING_PATIENCE` | 6 | Training |
| `LR_SCHEDULER_PATIENCE` | 3 | Training |
| `MAX_EPOCHS` | 30 | Training |

---

## Pipeline Summary (function order)

```
Raw Data
  → temporal_split_by_user (splitting.py)
  → fit_engagement_transform (splitting.py)
  → add_engagement_strength (splitting.py)
  → retain_train_supported_catalog (splitting.py)
  → add_contiguous_indices (dataset.py)
  → positive_history_map (dataset.py)
  → seen_item_map (dataset.py)
  → heldout_positive_map (dataset.py)
  → filter_targets_released_by_cutoff (dataset.py)
  → build_prefix_examples (dataset.py)
  → SteamSequenceTwoTowerDataset (dataset.py)
  → TwoTowerModel (model.py)
  → train_epoch (training.py)
  → padded_history_batch / available_candidate_indices (evaluation.py)
  → evaluate_filtered (evaluation.py)
  → evaluate_popularity_baseline (baselines.py)
  → evaluate_random_expected_baseline (baselines.py)
  → evaluate_content_centroid_baseline (baselines.py)
  → [training loop + eval, pipeline.py run_train_stage + run_eval_stage]
  → [HYBRID_ALPHA sweep, pipeline.py run_eval_stage]
  → [cold start, cold_start.py]
```

