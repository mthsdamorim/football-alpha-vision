"""The frozen GSR image-box assignment shared by TCC evaluation scripts."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = (a[k] for k in ("x_min", "y_min", "x_max", "y_max"))
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(tracks: list[dict], people: list[dict]) -> list[tuple[int, int]]:
    """Maximum cardinality, then IoU sum, then lexical source ID/ordinal."""
    if not tracks or not people:
        return []
    ordered = sorted(
        range(len(people)), key=lambda i: people[i]["source_annotation_id"]
    )
    ious = np.asarray(
        [
            [
                box_iou(tracks[j]["bbox"], people[i]["bbox_image"])
                for j in range(len(tracks))
            ]
            for i in ordered
        ]
    )
    count = len(ordered)
    cost = np.full((count, len(tracks) + count), 0.0)
    cost[:, : len(tracks)] = np.where(ious >= 0.5, -(count + 1 + ious), 1e6)

    def solve(locked: dict[int, int]) -> float:
        used = set(locked.values())
        free_rows = [i for i in range(count) if i not in locked]
        free_cols = [j for j in range(cost.shape[1]) if j not in used]
        fixed = sum(-cost[i, j] for i, j in locked.items())
        if not free_rows:
            return fixed
        rr, cc = linear_sum_assignment(cost[np.ix_(free_rows, free_cols)])
        return fixed + sum(
            -cost[free_rows[i], free_cols[j]] for i, j in zip(rr, cc, strict=True)
        )

    optimum = solve({})
    locked: dict[int, int] = {}
    for i in range(count):
        for j in range(cost.shape[1]):
            if j in locked.values() or cost[i, j] >= 1e6:
                continue
            candidate = {**locked, i: j}
            if abs(solve(candidate) - optimum) < 1e-10:
                locked = candidate
                break
        else:
            raise ValueError("matching objective could not be reconciled")
    return [
        (ordered[i], j)
        for i, j in locked.items()
        if j < len(tracks) and ious[i, j] >= 0.5
    ]
