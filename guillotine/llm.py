import os
import re
from functools import lru_cache
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"

@lru_cache(maxsize=1)
def _model(model_name: str):

    import transformers
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    return transformers.pipeline(model=model_name, dtype="auto", device_map="auto")


def _prompt(solution: str, clues: list[str]) -> str:
    lines = "\n".join(f"{i}. {c}" for i, c in enumerate(clues, 1))
    return (
        f"Sei un esperto del gioco televisivo La Ghigliottina.\n"
        f"La soluzione proposta è «{solution}».\n"
        f"I cinque indizi erano:\n{lines}\n\n"
        f"Per ciascun indizio, scrivi UNA frase in italiano che spiega come si "
        f"collega a «{solution}»: di solito insieme formano una parola composta, "
        f"una locuzione o un modo di dire.\n"
        f"Rispondi con un elenco puntato, un punto per indizio, senza preamboli. /no_think"
    )


def describe_solution(
    solution: str,
    clues: list[str],
    model_name: str = DEFAULT_MODEL,
    max_new_tokens: int = 400,
) -> str:
    messages = [{"role": "user", "content": _prompt(solution, clues)}]
    model = _model(model_name)
    text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    out = model(text, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
    gen = out[0]["generated_text"]
    return re.sub(r"^.*?</think>", "", gen, flags=re.DOTALL).strip()
