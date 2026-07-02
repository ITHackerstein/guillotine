from typing import Optional, Iterable
from collections import Counter, defaultdict
import math

MOD, BINOMIAL, DI, DA = "MOD", "BINOMIAL", "DI", "DA"
COMPOUND, IDIOM = "COMPOUND", "IDIOM"

PATTERN_RELATIONS = (MOD, BINOMIAL, DI, DA)
LEXICAL_RELATIONS = (COMPOUND, IDIOM)

_CONJ_LEMMAS = {"e", "ed"}
_DI_LEMMAS = {"di", "del", "dello", "della", "dell'", "dei", "degli", "delle"}
_DA_LEMMAS = {"da", "dal", "dallo", "dalla", "dall'", "dai", "dagli", "dalle"}

_NOMINAL = {"NOUN", "ADJ"}

Pairs = dict[str, Counter[tuple[str, str]]]

def _in_vocabulary(lemma: str, vocabulary: Optional[set[str]]) -> bool:
    return vocabulary is None or lemma in vocabulary

def _extract_typed_pairs(
    sentences: Iterable[list[tuple[str, str]]],
    vocabulary: Optional[set[str]] = None
) -> Pairs:
    pairs = {relation: Counter() for relation in PATTERN_RELATIONS}
    for sentence in sentences:
        n = len(sentence)
        for j in range(n):
            lemma_j, pos_j = sentence[j]
            if j + 2 < n:
                lemma_k, pos_k = sentence[j + 2]
                if _in_vocabulary(lemma_j, vocabulary) and _in_vocabulary(lemma_k, vocabulary) and pos_j in _NOMINAL and pos_k in _NOMINAL:
                    conn_lemma, conn_pos = sentence[j + 1]
                    if conn_lemma in _CONJ_LEMMAS and conn_pos == "CONJ":
                        pairs[BINOMIAL][(lemma_j, lemma_k)] += 1
                    if conn_lemma in _DI_LEMMAS and conn_pos == "ADP":
                        pairs[DI][(lemma_j, lemma_k)] += 1
                    if conn_lemma in _DA_LEMMAS and conn_pos == "ADP":
                        pairs[DA][(lemma_j, lemma_k)] += 1
            if j + 1 < n:
                lemma_k, pos_k = sentence[j + 1]
                if _in_vocabulary(lemma_j, vocabulary) and _in_vocabulary(lemma_k, vocabulary):
                    if pos_j == "NOUN" and pos_k == "ADJ":
                        pairs[MOD][(lemma_j, lemma_k)] += 1
                    if pos_j == "ADJ" and pos_k == "NOUN":
                        pairs[MOD][(lemma_k, lemma_j)] += 1
    return pairs


def _compute_ppmi(
    relation: Counter[tuple[str, str]],
    min_count: int,
) -> Iterable[tuple[str, str, float]]:
    total = sum(relation.values())
    if total == 0:
        return

    heads: dict[str, int] = defaultdict(int)
    tails: dict[str, int] = defaultdict(int)
    for (head, tail), count in relation.items():
        heads[head] += count
        tails[tail] += count
    for (head, tail), count in relation.items():
        if count < min_count:
            continue

        # NOTE: Derivation of the formula
        #         p(x, y) = count_xy / total
        #         p(x) = sum_y count_xy / total
        #         p(y) = sum_x count_xy / total
        #         pmi(x, y) = ln(p(x, y) / (p(x) * p(y)))
        #       The rest is a matter of simplifying the formula
        pmi = math.log(count * total / (heads[head] * tails[tail]))
        ppmi = pmi if pmi > 0.0 else 0.0
        yield (head, tail, ppmi)


def build_typed_triples(
    sentences: Iterable[list[tuple[str, str]]],
    vocabulary: Optional[set[str]] = None,
    min_count: int = 5
) -> dict[tuple[str, str, str], float]:
    pairs = _extract_typed_pairs(sentences, vocabulary)
    triples: dict[tuple[str, str, str], float] = {}
    for relation in PATTERN_RELATIONS:
        for head, tail, weight in _compute_ppmi(pairs[relation], min_count):
            triples[(head, relation, tail)] = weight
    return triples


def save_triples(
    triples: dict[tuple[str, str, str], float],
    path: str
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for triple, weight in triples.items():
            head, relation, tail = triple
            f.write(f"{head}\t{relation}\t{tail}\t{weight:.6g}\n")


def load_triples(path: str) -> dict[tuple[str, str, str], float]:
    triples: dict[tuple[str, str, str], float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            head, relation, tail, weight = line.split("\t")
            triples[(head, relation, tail)] = float(weight)
    return triples


def load_idiom_edges(path: str) -> list[tuple[str, str, int]]:
    edges: list[tuple[str, str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            a, b, count = line.split("\t")
            edges.append((a, b, int(count)))
    return edges


def merge_idiom_edges(
    triples: dict[tuple[str, str, str], float],
    edge_pairs: Iterable[tuple[str, str, int]],
    weight: float = 5.0
) -> dict[tuple[str, str, str], float]:
    merged = dict(triples)
    for a, b, count in edge_pairs:
        merged[(a, IDIOM, b)] = weight
    return merged

