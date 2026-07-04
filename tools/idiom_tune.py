import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

from guillotine.graph import load_triples, load_idiom_edges, merge_idiom_edges
from guillotine.eval import load_evalita, evaluate_solver
from guillotine.solver import ppmi_solver


# Per-worker shared state, populated once by _init_worker so the (potentially
# large) base-triples dict is loaded in each process instead of pickled per task.
_BASE: dict = {}
_IDIOM_EDGES: list = []
_GAMES: list = []
_COMBINE: str = "sum"


def _init_worker(base_triples: str, idiom: str, games: str, combine: str) -> None:
    global _BASE, _IDIOM_EDGES, _GAMES, _COMBINE
    _BASE = load_triples(base_triples)
    _IDIOM_EDGES = load_idiom_edges(idiom)
    _GAMES = load_evalita(games)
    _COMBINE = combine


def _eval_weight(w: float) -> tuple[float, dict, float]:
    triples = merge_idiom_edges(_BASE, _IDIOM_EDGES, weight=w)

    def solver(clues: list[str]) -> list[tuple[str, float]]:
        return ppmi_solver(triples, clues, combine=_COMBINE)

    t0 = time.time()
    metrics = evaluate_solver(solver, _GAMES)
    elapsed = time.time() - t0
    return w, metrics, elapsed


def main(args) -> None:
    # Load once in the parent purely to report sizes; workers load their own copies.
    base = load_triples(args.base_triples)
    print(f"[+] Loaded {len(base)} base triples (no IDIOM) from {args.base_triples}")

    idiom_edges = load_idiom_edges(args.idiom)
    print(f"[+] Loaded {len(idiom_edges)} idiom edges from {args.idiom}")

    games = load_evalita(args.games)
    print(f"[+] Loaded {len(games)} games from {args.games}")

    weights = [round(args.min_weight + i * args.step, 4)
               for i in range(int((args.max_weight - args.min_weight) / args.step) + 1)]

    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    workers = min(workers, len(weights))
    print(f"[.] Sweeping IDIOM weight over {len(weights)} values across {workers} "
          f"process(es): {weights}")

    print(f"\n{'weight':>8}  {'acc@1':>7}  {'mrr':>7}  {'found':>5}  {'sec':>5}")
    print("-" * 40)

    results: list[tuple[float, dict]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(args.base_triples, args.idiom, args.games, args.combine),
    ) as executor:
        # Preserve weight order in the printed table for readability.
        for w, metrics, elapsed in executor.map(_eval_weight, weights):
            results.append((w, metrics))
            print(f"{w:>8.2f}  {metrics['acc@1']:>7.4f}  {metrics['mrr']:>7.4f}  "
                  f"{metrics['found']:>5d}  {elapsed:>5.1f}")

    best_w, best_metrics = max(results, key=lambda x: (x[1]["acc@1"], x[1]["mrr"]))
    print("-" * 40)
    print(f"[+] Best weight: {best_w} with acc@1={best_metrics['acc@1']:.4f}, "
          f"mrr={best_metrics['mrr']:.4f}, found={best_metrics['found']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep the IDIOM edge weight on a games split.")
    parser.add_argument("--base-triples", type=str, default=".cache/base_triples.tsv",
                        help="Path to base triples (without IDIOM edges)")
    parser.add_argument("--idiom", type=str, default=".cache/mwe_idiom_edges.tsv",
                        help="Path to the idiom edge pairs")
    parser.add_argument("--games", type=str, default="dataset/train.json",
                        help="Path to the games split to tune on")
    parser.add_argument("--combine", type=str, default="sum",
                        choices=["max", "sum", "product", "coverage", "logsum"],
                        help="Solver combination method")
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=10.0)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel worker processes (0 = all CPU cores)")
    main(parser.parse_args())
