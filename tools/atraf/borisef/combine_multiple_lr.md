# combine_multiple_lr.py — Documentation

**File:** `tools/atraf/borisef/combine_multiple_lr.py`  
**Purpose:** Given a set of 2D line segments (each from a pair of detected keypoints with confidence scores), determine which direction they should all point, select the highest-quality segment as the "model", and flip the rest to match it.

---

## Data Model

Each segment is a Python tuple of two keypoints:

```python
((x1, y1, score1), (x2, y2, score2))
```

- `x1, y1`, `x2, y2` — 2D image coordinates (float or int)
- `score1, score2` — confidence scores in [0, 1]
- The segment direction is from point 1 → point 2 as given; this may need to be flipped.

---

## Public API

### `build_segment_table(segments, weight_score, weight_length, weight_consensus, length_threshold, score_threshold)`

**Purpose:** Analyse every segment and return a summary DataFrame used for model selection.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `segments` | — | List of `((x1,y1,s1),(x2,y2,s2))` |
| `weight_score` | `0.5` | Weight of normalised min-score in TotalScore |
| `weight_length` | `0.4` | Weight of normalised length in TotalScore |
| `weight_consensus` | `0.1` | Weight of normalised consensus in TotalScore |
| `length_threshold` | `5.0` | Segments shorter than this are marked invalid |
| `score_threshold` | `0.1` | Segments with minScore below this are marked invalid |

**Output columns:**

| Column | Type | Description |
|---|---|---|
| `scoreLeft` | float | Confidence of the spatially **left** endpoint (smaller x) |
| `scoreRight` | float | Confidence of the spatially **right** endpoint (larger x) |
| `minScore` | float | `min(scoreLeft, scoreRight)` — worst-case confidence |
| `Length` | float | Euclidean distance between the two endpoints |
| `Orientation` | float | Direction angle in **degrees [0, 360]**, computed as `atan2(dy, dx) % 360` from point 1 → point 2 |
| `valid` | int | `1` if `Length ≥ length_threshold` **and** `minScore ≥ score_threshold`, else `0` |
| `Consensus` | int | Count of all segments (including itself) whose direction has **positive cosine similarity** with this segment's direction vector |
| `normMinScore` | float | Min-max normalised `minScore` over all segments → [0, 1] |
| `normLength` | float | Min-max normalised `Length` over all segments → [0, 1] |
| `normConsensus` | float | Min-max normalised `Consensus` over all segments → [0, 1] |
| `TotalScore` | float | Weighted sum (see formula below); forced to `0.0` for invalid segments |

**TotalScore formula:**

```
TotalScore = weight_score     × normMinScore
           + weight_length    × normLength
           + weight_consensus × normConsensus
```

If `valid == 0` (segment fails either threshold), `TotalScore` is set to `0.0` regardless of the weighted sum.

**Consensus computation detail:**

For each pair (i, j), cosine similarity is computed from raw direction vectors:

```
cos_sim(i, j) = (v_i · v_j) / (|v_i| × |v_j|)
```

Segment j is counted toward segment i's consensus if `cos_sim(i, j) > 0`, i.e., their directions are within 90° of each other. This is more robust than a simple ±1 sign because it handles segments that are not perfectly anti-parallel.

**Min-max normalisation:**

```
norm(x) = (x - x_min) / (x_max - x_min)
```

If all values in a column are identical (x_max == x_min), the normalised column is set to 1.0 for all rows.

---

### `flip_to_model(segments, table)`

**Purpose:** Select the best segment as the "model" and flip all others to match its direction.

**Returns:**
- `corrected` — new segment list (same length as input); flipped segments have their two keypoints swapped: `(p1, p2)` becomes `(p2, p1)`.
- `model_idx` — integer index of the model segment in the list.

**Selection rule:**  
The segment with the **highest `TotalScore`** is chosen as the model. Because invalid segments have `TotalScore = 0.0`, they can never be selected as the model (unless all segments are invalid, in which case index 0 is returned by `idxmax()`).

**Flip rule:**  
For each other segment, its direction vector `v` is compared to the model direction vector `v_model` by cosine similarity. If `cos_sim < 0` (angle > 90° apart — opposite half-plane), the segment is flipped. The model segment itself is never flipped.

---

### `visualize_segments(segments_original, segments_corrected, table, model_idx, save_path=None)`

**Purpose:** Render a side-by-side matplotlib figure showing all segments before and after alignment.

**Layout:**
- Left subplot — **Before**: segments in their original orientations.
- Right subplot — **After**: segments after flipping to match the model.

**Visual encoding:**

| Element | Meaning |
|---|---|
| Green arrow | Segment already matched the model orientation |
| Red arrow | Segment was (or would be) flipped to match model |
| Gray faded arrow | Invalid segment (below length or score threshold) |
| Gold ★ star | The model segment (max TotalScore) |
| Index label | Small number at the arrow midpoint |
| Stats box at arrowhead | `nL`, `nS`, `nC`, `T` values for that segment (see below) |

**Per-segment stats annotation** (placed at the arrowhead tip, aligned away from the arrow body):

```
nL=x.xx  nS=x.xx
nC=x.xx  T=x.xx
```

Where `nL` = normLength, `nS` = normMinScore, `nC` = normConsensus, `T` = TotalScore.

The y-axis is inverted (`invert_yaxis()`) so the coordinate origin matches image (pixel) convention — y increases downward.

**`save_path`:** if provided, saves the figure as a PNG at that path instead of calling `plt.show()`.

---

## Full Pipeline (as in `main()`)

```python
# 1. Build table
table = build_segment_table(segments,
                             weight_score=0.5,
                             weight_length=0.4,
                             weight_consensus=0.1,
                             length_threshold=5.0,
                             score_threshold=0.1)

# 2. Select model and align
corrected, model_idx = flip_to_model(segments, table)

# 3. Visualise
visualize_segments(segments, corrected, table, model_idx, save_path='out.png')
```

---

## Demo Data (`main()`)

Ten segments designed to cover all cases:

| # | Description | valid | Expected |
|---|---|---|---|
| 0 | Long, high score, →~2° | 1 | **MODEL** |
| 1 | Long, high score, ←~178° | 1 | FLIPPED |
| 2 | Medium, medium score, →~7° | 1 | kept |
| 3 | Medium, score below threshold (0.07) | 0 | INVALID |
| 4 | Short (len≈3), reversed | 0 | INVALID |
| 5 | Long, medium score, ←~178° | 1 | FLIPPED |
| 6 | Long, high score, →~2° | 1 | kept |
| 7 | Medium, low-but-valid score, →~4° | 1 | kept |
| 8 | Short (len≈3), correct | 0 | INVALID |
| 9 | Long, medium-high score, →~358° | 1 | kept |

---

## Running

```bash
python3 tools/atraf/borisef/combine_multiple_lr.py
```

Prints the full 10-column DataFrame to stdout, then opens a matplotlib window (or saves a PNG if `save_path` is set).

**Dependencies:** `numpy`, `pandas`, `matplotlib` — no mmpose imports required.
