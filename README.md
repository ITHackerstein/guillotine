# Guillotine

A solver for *La Ghigliottina*, the final game of the Italian TV show *L'Eredità*, in
which five clue words are given and the player must guess the single word that
connects to all five (usually as a common multi-word expression or collocation).

The solver builds a *typed*, weighted word graph from the PAISÀ corpus, where edge
types encode MWE-bearing syntactic patterns (noun–adjective, *X e Y*, *X di Y*,
*X da Y*) and edge weights are pattern-conditioned PPMI. Curated `IDIOM` edges
mined from Italian *polirematiche* and proverb lists are folded in on top. A
flat PPMI-sum solver over this graph reaches **accuracy@1 0.34 / MRR 0.39** on
the EVALITA 2018 NLP4FUN test set. Optionally, an LLM can be used to describe
how each clue relates to the predicted solution.

See `docs/report.pdf` for the full write-up.

## Dependencies

- Python `>=3.13,<3.14` (torch has no `cp314` wheels yet).
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.
- [`typst`](https://typst.app/) — only needed to rebuild the documents under `docs/`.

Runtime Python dependencies are declared in `pyproject.toml` and split into
groups:

- `tools` — `spacy` and `click`, used by the scripts under `tools/` to mine
  `IDIOM` edges from the *polirematiche* / proverb lists.
- `llm` — `transformers`, `accelerate`, `evaluate`, `sacrebleu`, `rouge-score`,
  `bert-score`, used by the optional LLM-based description feature and by the
  description evaluation.
- `llm-cpu` / `llm-nvidia` / `llm-amd` — mutually exclusive backend groups
  that pull `torch` (and `pytorch-triton-rocm` on AMD) from the matching
  PyTorch index. Pick exactly one for your hardware.

### Data

The following data directories are expected at the repository root and are not
tracked in git:

- `paisa/` — the PAISÀ corpus.
- `polirematiche_proverbi/` — De Mauro *polirematiche*, proverbs and Wiktionary
  MWE lists used to build `IDIOM` edges.
- `dataset/` — EVALITA 2018 NLP4FUN games (`train.json`, `test.json`).

## Environment setup

Install `uv` (see the [uv docs](https://docs.astral.sh/uv/getting-started/installation/)),
then sync the environment. Pick exactly one backend group for `torch`:

```sh
# CPU only
uv sync --group llm --group llm-cpu

# NVIDIA (CUDA 12.4)
uv sync --group llm --group llm-nvidia

# AMD (ROCm 6.3)
uv sync --group llm --group llm-amd
```

If you also need the MWE mining scripts under `tools/`, add `--group tools`:

```sh
uv sync --group tools --group llm --group llm-cpu
```

Mine the `IDIOM` edges once (writes to `.cache/mwe_idiom_edges.tsv`):

```sh
uv run python -m tools.mwe_edges
```

The first solver run will build the typed triples from PAISÀ and cache them at
`.cache/triples.tsv`; subsequent runs reuse the cache.

## Running the CLI

The CLI is exposed as the `guillotine` module and has two subcommands,
`evaluate` and `solve`.

### Evaluate on the EVALITA test set

Default paths (`paisa/…`, `.cache/…`, `dataset/test.json`) are picked up
automatically:

```sh
uv run python -m guillotine.cli evaluate
```

Try a different score-combination strategy:

```sh
uv run python -m guillotine.cli evaluate --combine logsum
```

Evaluate both the solution ranking *and* the LLM-generated descriptions
(requires the `llm` group):

```sh
uv run python -m guillotine.cli evaluate --describe
```

Cap corpus loading for a quick smoke test:

```sh
uv run python -m guillotine.cli evaluate --max-sentences 100000
```

### Solve a single game

Pass exactly five comma-separated clues:

```sh
uv run python -m guillotine.cli solve --clues "carta,penna,banco,voto,maestra"
```

Also print an LLM explanation of how each clue relates to the predicted
solution:

```sh
uv run python -m guillotine.cli solve \
    --clues "carta,penna,banco,voto,maestra" \
    --describe
```

Run `uv run python -m guillotine.cli {evaluate,solve} --help` for the full
list of options (vocabulary size, min frequency / count thresholds, alternate
corpus or triples files, etc.).
