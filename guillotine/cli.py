import argparse
import time
import os
from .corpus import load_corpus, load_vocabulary
from .graph import load_triples, save_triples, build_typed_triples, load_idiom_edges, merge_idiom_edges
from .eval import load_evalita, evaluate_solver, format_metrics
from .solver import ppmi_solver

DEFAULT_CORPUS =  "paisa/paisa.annotated.CoNLL.utf8.gz"
DEFAULT_FREQ = "paisa/lemma-WITHOUTnumberssymbols-frequencies-paisa.txt.gz"
DEFAULT_IDIOM = ".cache/mwe_idiom_edges.tsv"
DEFAULT_TRIPLES = ".cache/triples.tsv"
DEFAULT_GAMES = "dataset/test.json"

def _get_triples(args):
    if os.path.exists(args.triples):
        print(f"[.] Existing triples file found at {args.triples}, loading...")
        return load_triples(args.triples)

    sentences = load_corpus(args.corpus, max_sentences=args.max_sentences)
    print(f"[+] Loaded sentences from {args.corpus}")

    vocabulary = load_vocabulary(args.freq, top_n=args.vocab_top_n, min_frequency=args.min_frequency)
    print(f"[+] Loaded vocabulary with {len(vocabulary)} lemmas from {args.freq}")

    t0 = time.time()
    print(f"[.] Building typed triples from corpus...")
    triples = build_typed_triples(sentences, vocabulary=vocabulary, min_count=args.min_count)
    print(f"[+] Built {len(triples)} typed triples in {time.time() - t0:.1f} seconds")

    idiom_edges = load_idiom_edges(args.idiom)
    print(f"[+] Loaded {len(idiom_edges)} idiom edges from {args.idiom}")

    triples = merge_idiom_edges(triples, idiom_edges)
    print(f"[+] Merged idiom edges into triples, now {len(triples)} triples")

    save_triples(triples, args.triples)
    print(f"[+] Saved triples to {args.triples}")

    return triples


def evaluate(args):
    triples = _get_triples(args)
    games = load_evalita(args.games)
    print(f"[+] Loaded {len(games)} games from {args.games}")

    def solver(clues: list[str]) -> list[tuple[str, float]]:
        return ppmi_solver(triples, clues, combine=args.combine)

    print(f"[.] Evaluating solver with {args.combine} combination method...")
    metrics = evaluate_solver(solver, games)
    print(f"[+] Solution evaluation metrics:\n    {format_metrics(metrics)}")

    if args.describe:
        from .llm import describe_solution
        from .eval import evaluate_descriptions
        print(f"[.] Collecting predicted descriptions for each solution...")
        actual = [game.description for game in games]
        predicted = []
        t0 = time.time()
        for i, game in enumerate(games, 1):
            solution, _ = solver(game.clues)[0]
            predicted.append(describe_solution(solution, game.clues))
            if i % 5 == 0:
                print(f"[.] Processed {i}/{len(games)} games. {(time.time() - t0) / i:.1f} games/s")
        print(f"[+] Collected predicted descriptions for {len(games)} games in {time.time() - t0:.1f} seconds")

        print(f"[.] Evaluating description quality...")
        metrics = evaluate_descriptions(actual, predicted)
        print(f"[+] Description evaluation metrics:\n    {format_metrics(metrics)}")


def solve(args):
    triples = _get_triples(args)
    clues = args.clues.split(",") if args.clues else []
    if len(clues) != 5:
        raise ValueError("Exactly five clues must be provided, separated by commas.")

    clues = [clue.strip().lower() for clue in clues]
    ranking = ppmi_solver(triples, clues, combine=args.combine)
    solution, score = ranking[0]
    print(f"[+] Top solution: {solution} with score {score:.4f}")
    if args.describe:
        from .llm import describe_solution
        explanation = describe_solution(solution, clues)
        print(f"[+] Explanation:\n{explanation}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="guillotine", description="Solve La Ghigliottina via a typed PPMI graph.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the solver on a set of games")
    evaluate_parser.add_argument("--corpus", type=str, default=DEFAULT_CORPUS, help="Path to the corpus file")
    evaluate_parser.add_argument("--freq", type=str, default=DEFAULT_FREQ, help="Path to the lemma frequency file")
    evaluate_parser.add_argument("--idiom", type=str, default=DEFAULT_IDIOM, help="Path to the idiom triples")
    evaluate_parser.add_argument("--triples", type=str, default=DEFAULT_TRIPLES, help="Path to the triples file")
    evaluate_parser.add_argument("--vocab-top-n", type=int, default=50_000, help="Top N vocabulary lemmas to consider")
    evaluate_parser.add_argument("--max-sentences", type=int, default=None, help="Maximum number of sentences to load from the corpus")
    evaluate_parser.add_argument("--min-frequency", type=int, default=5, help="Minimum frequency for lemmas to be included in the vocabulary")
    evaluate_parser.add_argument("--min-count", type=int, default=5, help="Minimum count for pairs to be included in the triples")
    evaluate_parser.add_argument("--combine", type=str, choices=["max", "sum", "product", "coverage", "logsum"], default="sum", help="Method to combine clue scores")
    evaluate_parser.add_argument("--describe", action="store_true", help="Use LLM to describe how each clue relates to the solution")
    evaluate_parser.add_argument("--games", type=str, default=DEFAULT_GAMES, help="Path to the games file")
    evaluate_parser.set_defaults(func=evaluate)

    solve_parser = subparsers.add_parser("solve", help="Solve a single game with given clues")
    solve_parser.add_argument("--corpus", type=str, default=DEFAULT_CORPUS, help="Path to the corpus file")
    solve_parser.add_argument("--freq", type=str, default=DEFAULT_FREQ, help="Path to the lemma frequency file")
    solve_parser.add_argument("--idiom", type=str, default=DEFAULT_IDIOM, help="Path to the idiom triples")
    solve_parser.add_argument("--triples", type=str, default=DEFAULT_TRIPLES, help="Path to the triples file")
    solve_parser.add_argument("--vocab-top-n", type=int, default=50_000, help="Top N vocabulary lemmas to consider")
    solve_parser.add_argument("--max-sentences", type=int, default=None, help="Maximum number of sentences to load from the corpus")
    solve_parser.add_argument("--min-frequency", type=int, default=5, help="Minimum frequency for lemmas to be included in the vocabulary")
    solve_parser.add_argument("--min-count", type=int, default=5, help="Minimum count for pairs to be included in the triples")
    solve_parser.add_argument("--combine", type=str, choices=["max", "sum", "product", "coverage", "logsum"], default="sum", help="Method to combine clue scores")
    solve_parser.add_argument("--describe", action="store_true", help="Use LLM to describe how each clue relates to the solution")
    solve_parser.add_argument("--clues", type=str, help="Comma-separated list of five clues for the game")
    solve_parser.set_defaults(func=solve)

    args = parser.parse_args()
    args.func(args)
