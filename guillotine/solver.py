import math
from collections import defaultdict
from typing import Optional

def _compute_max_adjacency(
    weights: dict[tuple[str, str, str], float]
) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for (head, _, tail), weight in weights.items():
        if weight > adjacency[head].get(tail, -math.inf):
            adjacency[head][tail] = weight
        if weight > adjacency[tail].get(head, -math.inf):
            adjacency[tail][head] = weight
    return adjacency


def ppmi_solver(
    weights: dict[tuple[str, str, str], float],
    clues: list[str],
    combine: str = "sum"
) -> list[tuple[str, float]]:
    adjacency = _compute_max_adjacency(weights)
    clues = [c.lower() for c in clues]
    candidates: set[str] = set()
    for c in clues:
        candidates.update(adjacency.get(c, {}).keys())
    candidates -= set(clues)

    scores: list[tuple[str, float]] = []
    for candidate in candidates:
        per_clue_scores = [adjacency.get(clue, {}).get(candidate, 0.0) for clue in clues]
        if combine == "max":
            score = max(per_clue_scores)
        elif combine == "sum":
            score = sum(per_clue_scores)
        elif combine == "product":
            score = math.prod(per_clue_scores)
        elif combine == "coverage":
            covered = sum(1 for v in per_clue_scores if v > 0.0)
            score = covered * 1e6 + sum(per_clue_scores)
        elif combine == "logsum":
            if any(v <= 0.0 for v in per_clue_scores):
                continue
            score = sum(math.log1p(v) for v in per_clue_scores)
        else:
            raise ValueError(f"Unknown combine method: {combine}")
        scores.append((candidate, score))

    return sorted(scores, key=lambda x: -x[1])
