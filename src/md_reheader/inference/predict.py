from md_reheader.data.format import (
    SYSTEM_PROMPT,
    parse_headings_output,
)
from md_reheader.models import Heading


def predict_heading_levels(
    md_text: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> list[Heading]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": md_text},
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return parse_headings_output(response)
