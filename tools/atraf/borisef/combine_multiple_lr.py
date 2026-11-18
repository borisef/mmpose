"""
Standalone utility: analyse orientation of 2D line segments and align them all
to match the highest-scoring "model" segment.

Each segment is ((x1, y1, score1), (x2, y2, score2)).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Step 1 – Build the preparation table (raw + normalised + TotalScore)
# ---------------------------------------------------------------------------

def build_segment_table(
    segments,
    weight_score=0.5,
    weight_length=0.4,
    weight_consensus=0.1,
    length_threshold=5.0,
    score_threshold=0.1,
):
    """
    Parameters
    ----------
    segments : list of ((x1,y1,s1), (x2,y2,s2))

    Returns
    -------
    pd.DataFrame with columns:
        scoreLeft, scoreRight, minScore, Length, Orientation [deg 0-360],
        valid, Consensus, normMinScore, normLength, normConsensus, TotalScore
    """
    rows = []
    vectors = []   # direction vector for each segment as given

    for (x1, y1, s1), (x2, y2, s2) in segments:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        angle_deg = float(np.degrees(np.arctan2(dy, dx)) % 360)

        # scoreLeft / scoreRight = scores of the spatially left/right endpoints
        if x1 <= x2:
            score_left, score_right = s1, s2
        else:
            score_left, score_right = s2, s1

        min_score = min(score_left, score_right)
        valid = int(length >= length_threshold and min_score >= score_threshold)

        rows.append({
            'scoreLeft':   score_left,
            'scoreRight':  score_right,
            'minScore':    min_score,
            'Length':      round(length, 2),
            'Orientation': round(angle_deg, 1),
            'valid':       valid,
        })
        vectors.append(np.array([dx, dy], dtype=float))

    df = pd.DataFrame(rows)

    # Consensus via cosine similarity: count segments with cos_sim > 0 w.r.t. me
    norms = [np.linalg.norm(v) for v in vectors]
    consensus = []
    for i, vi in enumerate(vectors):
        count = 0
        for j, vj in enumerate(vectors):
            if norms[i] > 0 and norms[j] > 0:
                cos_sim = np.dot(vi, vj) / (norms[i] * norms[j])
                if cos_sim > 0:
                    count += 1
        consensus.append(count)
    df['Consensus'] = consensus

    # Normalise to [0, 1] with min-max scaling
    def minmax(col):
        lo, hi = col.min(), col.max()
        if hi == lo:
            return pd.Series(np.ones(len(col)), index=col.index)
        return (col - lo) / (hi - lo)

    df['normMinScore']  = minmax(df['minScore'])
    df['normLength']    = minmax(df['Length'])
    df['normConsensus'] = minmax(df['Consensus'])

    # Weighted TotalScore
    df['TotalScore'] = (
        weight_score     * df['normMinScore'] +
        weight_length    * df['normLength']   +
        weight_consensus * df['normConsensus']
    )

    # Zero out invalid segments
    df.loc[df['valid'] == 0, 'TotalScore'] = 0.0

    return df


# ---------------------------------------------------------------------------
# Step 2 – Select model segment and align all others to its direction
# ---------------------------------------------------------------------------

def flip_to_model(segments, table):
    """
    The segment with the highest TotalScore is the model; its direction is kept.
    Every other segment is flipped if its cosine similarity with the model < 0.

    Returns
    -------
    corrected : list of segments
    model_idx : int
    """
    model_idx = int(table['TotalScore'].idxmax())
    (mx1, my1, _), (mx2, my2, _) = segments[model_idx]
    model_vec = np.array([float(mx2 - mx1), float(my2 - my1)])
    model_norm = np.linalg.norm(model_vec)

    corrected = []
    for seg in segments:
        (x1, y1, _), (x2, y2, _) = seg
        v = np.array([float(x2 - x1), float(y2 - y1)])
        v_norm = np.linalg.norm(v)
        if model_norm > 0 and v_norm > 0 and np.dot(model_vec, v) / (model_norm * v_norm) < 0:
            corrected.append((seg[1], seg[0]))   # flip
        else:
            corrected.append(seg)

    return corrected, model_idx


# ---------------------------------------------------------------------------
# Step 3 – Visualise
# ---------------------------------------------------------------------------

def _cos_sim_with_model(seg, model_vec, model_norm):
    (x1, y1, _), (x2, y2, _) = seg
    v = np.array([float(x2 - x1), float(y2 - y1)])
    v_norm = np.linalg.norm(v)
    if model_norm > 0 and v_norm > 0:
        return float(np.dot(model_vec, v) / (model_norm * v_norm))
    return 1.0   # treat degenerate as aligned


def _draw_arrow(ax, x1, y1, x2, y2, color, alpha=1.0, label=None, stats=None):
    ax.annotate(
        '',
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=1.8, alpha=alpha),
    )
    if label is not None:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, str(label), fontsize=7, ha='center', va='bottom',
                color=color, alpha=alpha)
    if stats is not None:
        norm_length, norm_score, norm_consensus, total_score = stats
        txt = (f'nL={norm_length:.2f}  nS={norm_score:.2f}\n'
               f'nC={norm_consensus:.2f}  T={total_score:.2f}')
        # place text at the arrowhead tip; align away from the arrow body
        ha = 'left' if x2 >= x1 else 'right'
        ax.text(x2, y2, txt, fontsize=5.5, ha=ha, va='center',
                color=color, alpha=alpha,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.65 * alpha))


def visualize_segments(segments_original, segments_corrected, table, model_idx,
                       save_path=None):
    """
    Two side-by-side subplots: Before and After orientation correction.
      Green  = matches model orientation (or is the model).
      Red    = was / would be flipped.
      Gray   = invalid (below threshold), drawn faded.
      Gold * = model segment (max TotalScore).
    """
    (mx1, my1, _), (mx2, my2, _) = segments_original[model_idx]
    model_vec = np.array([float(mx2 - mx1), float(my2 - my1)])
    model_norm = np.linalg.norm(model_vec)

    is_flipped = [
        _cos_sim_with_model(seg, model_vec, model_norm) < 0
        for seg in segments_original
    ]
    is_valid = table['valid'].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for ax, segs, title in zip(
        axes,
        [segments_original, segments_corrected],
        ['Before (original)', 'After (model-aligned)'],
    ):
        for i, (seg, flipped, valid) in enumerate(zip(segs, is_flipped, is_valid)):
            (x1, y1, _), (x2, y2, _) = seg
            if not valid:
                color, alpha = 'gray', 0.35
            elif flipped:
                color, alpha = 'red', 1.0
            else:
                color, alpha = 'green', 1.0
            seg_stats = (
                table.loc[i, 'normLength'],
                table.loc[i, 'normMinScore'],
                table.loc[i, 'normConsensus'],
                table.loc[i, 'TotalScore'],
            )
            _draw_arrow(ax, x1, y1, x2, y2, color=color, alpha=alpha,
                        label=i, stats=seg_stats)
            ax.plot([x1, x2], [y1, y2], '.', color=color, markersize=5, alpha=alpha)

        # mark the model segment
        (gx1, gy1, _), (gx2, gy2, _) = segs[model_idx]
        ax.plot((gx1 + gx2) / 2, (gy1 + gy2) / 2,
                '*', color='gold', markersize=14, zorder=5)

        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    green_patch = mpatches.Patch(color='green', label='matches model orientation')
    red_patch   = mpatches.Patch(color='red',   label='was / would be flipped')
    gray_patch  = mpatches.Patch(color='gray',  label='invalid (below threshold)')
    gold_patch  = mpatches.Patch(color='gold',  label='model segment (max TotalScore)')
    fig.legend(handles=[green_patch, red_patch, gray_patch, gold_patch],
               loc='lower center', ncol=4)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f'Figure saved to {save_path}')
    else:
        plt.show()


# ---------------------------------------------------------------------------
# main – demo with varied lengths, scores and orientations
# ---------------------------------------------------------------------------

def main():
    # Segments are mostly horizontal (left→right), a few reversed (←),
    # two invalid by score, two invalid by length.
    segments = [
        # long, high score, correct →  (~2°)
        ((10,  10, 0.95), (90,  13, 0.88)),
        # long, high score, reversed ← (~178°)
        ((95,  25, 0.82), (15,  28, 0.75)),
        # medium, medium score, correct → (~7°)
        ((20,  42, 0.65), (70,  48, 0.70)),
        # medium, BELOW score threshold, correct →
        ((25,  57, 0.08), (75,  59, 0.07)),
        # SHORT (below length threshold), reversed ←
        ((62,  72, 0.70), (59,  73, 0.65)),
        # long, medium score, reversed ← (~178°)
        ((88,  87, 0.60), (12,  90, 0.55)),
        # long, high score, correct → (~2°)
        ((15, 102, 0.90), (85, 105, 0.85)),
        # medium, low-but-valid score, correct → (~4°)
        ((30, 117, 0.15), (70, 120, 0.12)),
        # SHORT (below length threshold), correct →
        ((42, 132, 0.80), (45, 133, 0.78)),
        # long, medium-high score, correct → (slight upward, ~358°)
        (( 5, 147, 0.78), (80, 144, 0.72)),
    ]

    pd.set_option('display.float_format', '{:.3f}'.format)
    table = build_segment_table(segments)

    print('=== Segment table ===')
    print(table.to_string(index=True))
    print()

    corrected, model_idx = flip_to_model(segments, table)

    print(f'=== Model segment: #{model_idx} '
          f'(TotalScore={table.loc[model_idx, "TotalScore"]:.3f}, '
          f'Orientation={table.loc[model_idx, "Orientation"]}°) ===')
    for i, (orig, corr) in enumerate(zip(segments, corrected)):
        if i == model_idx:
            tag = '[MODEL]'
        elif not table.loc[i, 'valid']:
            tag = 'INVALID'
        elif orig != corr:
            tag = 'FLIPPED'
        else:
            tag = 'kept   '
        print(f'  Segment {i}: {tag}')
    print()

    visualize_segments(segments, corrected, table, model_idx)


if __name__ == '__main__':
    main()
