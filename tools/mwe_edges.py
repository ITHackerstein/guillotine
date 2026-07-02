import os
from typing import Iterable
from guillotine.corpus import load_vocabulary
import time
import itertools
import spacy

MWE_DIR = "polirematiche_proverbi"
PAISA_FREQ = "paisa/lemma-WITHOUTnumberssymbols-frequencies-paisa.txt.gz"
NOMINAL_POS = {"NOUN", "ADJ", "PROPN"}

def _phrases() -> Iterable[str]:
    with open(os.path.join(MWE_DIR, "demauro.poli"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.replace(".", "").isdigit():
                continue
            yield line
    with open(os.path.join(MWE_DIR, "polirematiche"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            _pos, phrase = line.split("\t", 1)
            phrase = phrase.strip()
            if phrase:
                yield phrase
    with open(os.path.join(MWE_DIR, "proverbi"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def main(args) -> None:
    vocabulary = load_vocabulary(PAISA_FREQ, top_n=args.vocab_top_n)
    print(f"[+] Loaded vocabulary with {len(vocabulary)} lemmas")

    phrases = list(_phrases())
    print(f"[+] Loaded {len(phrases)} phrases from {MWE_DIR}")

    nlp = spacy.load("it_core_news_sm", disable=["ner", "parser"])
    pair_counts: dict[tuple[str, str], int] = {}
    t0 = time.time()
    for i, doc in enumerate(nlp.pipe(phrases, batch_size=512, n_process=args.worker_count), 1):
        lemmas = set()
        for token in doc:
            if token.pos_ not in NOMINAL_POS:
                continue

            lemma = (token.lemma_ or token.text).lower()
            if " " in lemma:
                lemma = lemma.split(" ", 1)[0] # NOTE: The maxsplit=1 is for efficiency
            if lemma in vocabulary:
                lemmas.add(lemma)

        if len(lemmas) >= 2:
            for a, b in itertools.combinations(sorted(lemmas), 2):
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

        if i % 10_000 == 0:
            print(f"[.] {i} / {len(phrases)} phrases tagged {i/(time.time() - t0):.0f}/s")

    print(f"[+] Found {len(pair_counts)} distinct lemma pairs. Took {time.time() - t0:.0f}s")
    with open(args.output, "w", encoding="utf-8") as f:
        for (a, b), count in sorted(pair_counts.items(), key=lambda x: -x[1]):
            f.write(f"{a}\t{b}\t{count}\n")
    print(f"[+] Saved lemma pairs to {args.output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mine lemma pairs from multi-word expressions and idioms")
    parser.add_argument("--vocab-top-n", type=int, default=50_000, help="Top N vocabulary lemmas to consider")
    parser.add_argument("--worker-count", type=int, default=8, help="Number of workers for spaCy")
    parser.add_argument("--output", type=str, default=".cache/mwe_idiom_edges.tsv", help="Output file for lemma pairs")

    args = parser.parse_args()
    main(args)
