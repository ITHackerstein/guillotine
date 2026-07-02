import gzip
import lzma
import io
from typing import Iterator, Optional

def _open_compressed_or_text(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.GzipFile(path, "rb"), encoding="utf-8", errors="replace")
    if path.endswith(".xz"):
        return io.TextIOWrapper(lzma.LZMAFile(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


NOUN, ADJ, VERB, ADV, CONJ, ADP, DET, NUM, PRON, PUNCT, OTHER = (
    "NOUN", "ADJ", "VERB", "ADV", "CONJ", "ADP", "DET", "NUM", "PRON", "PUNCT", "X",
)

_TANL = {
    "S": NOUN, "SP": NOUN,
    "A": ADJ, "AP": ADJ,
    "V": VERB, "VA": VERB, "VM": VERB,
    "B": ADV, "BN": ADV,
    "C": CONJ, "CC": CONJ, "CS": CONJ,
    "E": ADP, "EA": ADP,
    "D": DET, "DI": DET, "DD": DET, "DQ": DET, "DR": DET, "DE": DET,
    "R": DET, "RI": DET, "RD": DET,
    "N": NUM, "NO": NUM,
    "P": PRON, "PC": PRON, "PR": PRON, "PI": PRON, "PP": PRON, "PD": PRON,
    "PQ": PRON, "PE": PRON,
    "F": PUNCT, "FF": PUNCT, "FB": PUNCT, "FC": PUNCT, "FS": PUNCT,
    "I": OTHER, "T": DET, "X": OTHER,
}

_UD = {
    "NOUN": NOUN, "PROPN": NOUN,
    "ADJ": ADJ,
    "VERB": VERB, "AUX": VERB,
    "ADV": ADV,
    "CCONJ": CONJ, "SCONJ": CONJ, "CONJ": CONJ,
    "ADP": ADP,
    "DET": DET,
    "NUM": NUM,
    "PRON": PRON,
    "PUNCT": PUNCT, "SYM": PUNCT,
    "INTJ": OTHER, "PART": OTHER, "X": OTHER,
}

def _map_pos(tag: str) -> str:
    if not tag:
        return OTHER
    if tag in _TANL:
        return _TANL[tag]
    if tag in _UD:
        return _UD[tag]
    return _TANL.get(tag[0], OTHER)


def load_corpus(
    path: str,
    max_sentences: Optional[int] = None,
    lower: bool = True
) -> Iterator[list[tuple[str, str]]]:
    sentence: list[tuple[str, str]] = []
    sentence_count = 0
    with _open_compressed_or_text(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if len(sentence) > 0:
                    yield sentence
                    sentence_count += 1
                    if max_sentences is not None and sentence_count >= max_sentences:
                        return
                    sentence = []
                continue

            # Skips comments and <text></text> tags
            if line[0] in "<#":
                continue

            columns = line.split("\t")
            # Not a token line
            if len(columns) < 4:
                continue

            token_id = columns[0]
            # Skips range tokens and empty tokens
            if "-" in token_id or "." in token_id:
                continue

            lemma = columns[2]
            # If the LEMMA is missing we fall back to the FORM column
            if lemma == "_" or not lemma:
                lemma = columns[1]

            if lower:
                lemma = lemma.lower()

            sentence.append((lemma, _map_pos(columns[3])))

        if len(sentence) > 0 and (max_sentences is None or sentence_count < max_sentences):
            yield sentence


def load_vocabulary(
    path: str,
    top_n: int = 50_000,
    min_frequency: int = 5
) -> set[str]:
    with _open_compressed_or_text(path) as f:
        entries: list[tuple[str, int]] = []
        for line in f:
            line = line.rstrip("\n")
            if not line or line[0] == "#":
                continue

            columns = line.split(",")
            if len(columns) < 2:
                continue

            word, frequency = columns[0], int(columns[1])
            if frequency >= min_frequency:
                entries.append((word, frequency))

        entries = sorted(entries, key=lambda entry: -entry[1])
        return {lemma for lemma, _ in entries[:top_n]}
