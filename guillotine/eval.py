from dataclasses import dataclass
from typing import Callable, Optional
import json

@dataclass
class Game:
    clues: list[str]
    solution: str
    description: str

def load_evalita(path: str) -> list[Game]:
    games: list[Game] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            game_raw = json.loads(line)
            games.append(Game(
                clues=[clue.strip().lower() for clue in [game_raw[f"hint{i + 1}"] for i in range(5)]],
                solution=game_raw["sol"].strip().lower(),
                description=game_raw["desc"].strip()
            ))
    return games

Solver = Callable[[list[str]], list[tuple[str, float]]]

def _ranking_of(ranking: list[tuple[str, float]], solution: str) -> Optional[int]:
    for i, (word, _) in enumerate(ranking):
        if word == solution:
            return i + 1
    return None

def evaluate_solver(
    solver: Solver,
    games: list[Game],
    top_ks: tuple[int, ...] = (1, 5, 10, 100),
    verbose: bool = False
) -> dict:
    max_k = max(top_ks)
    rr = 0.0
    hits_per_k = {k: 0 for k in top_ks}
    found = 0
    for game in games:
        ranking = solver(game.clues)[:max_k]
        rank = _ranking_of(ranking, game.solution)
        if rank is not None:
            rr += 1.0 / rank
            found += 1
            for k in top_ks:
                if rank <= k:
                    hits_per_k[k] += 1

        if verbose:
            top_words = ", ".join(word for word, _ in ranking[:5])
            mark = f"#{rank}" if rank is not None else "not found"
            print(f"Clues: {game.clues} | Solution: {game.solution} | Top words: {top_words} | Rank: {mark}")

    metrics = {
        "n": len(games),
        "found": found,
        "mrr": rr / len(games),
    }

    for k in top_ks:
        metrics[f"acc@{k}"] = hits_per_k[k] / len(games)

    return metrics


def evaluate_descriptions(actual: list[str], predicted: list[str]) -> dict:
    import evaluate

    metrics = {}

    bleu = evaluate.load("sacrebleu")
    metrics["bleu"] = bleu.compute(predictions=predicted, references=[[r] for r in actual])["score"]

    rouge = evaluate.load("rouge")
    metrics.update(rouge.compute(predictions=predicted, references=actual, use_stemmer=False))

    bertscore = evaluate.load("bertscore")
    r = bertscore.compute(predictions=predicted, references=actual, lang="it")
    metrics["bertscore"] = sum(r["f1"]) / len(r["f1"])

    return metrics


def format_metrics(metrics: dict) -> str:
    return " ".join(
        f"{metric} " +
        (f"{value}" if isinstance(value, int) else f"{value:.6f}")
        for metric, value in metrics.items()
    )
