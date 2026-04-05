import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from md_reheader.data.apply import apply_levels
from md_reheader.data.extract import extract_headings
from md_reheader.data.format import SYSTEM_PROMPT, parse_levels_from_output
from md_reheader.data.strip import strip_document


def load_model(
    model_path: str,
    device: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device_map = {"": device} if device != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def predict_heading_levels(
    md_text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int = 4096,
) -> list[int]:
    stripped = strip_document(md_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": stripped},
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return parse_levels_from_output(response)


def reheader_document(
    md_text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """Full pipeline: predict heading levels and apply them to the original document."""
    headings = extract_headings(md_text)
    if not headings:
        return md_text

    levels = predict_heading_levels(md_text, model, tokenizer)

    # Align prediction count with actual headings
    if len(levels) < len(headings):
        levels.extend([headings[i].level for i in range(len(levels), len(headings))])
    elif len(levels) > len(headings):
        levels = levels[: len(headings)]

    return apply_levels(md_text, levels)
