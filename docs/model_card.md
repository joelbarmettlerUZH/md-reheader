---
license: apache-2.0
language:
  - en
tags:
  - markdown
  - document-structure
  - heading-prediction
  - text-classification
  - long-context
base_model: Qwen/Qwen3-0.6B
metrics:
  - accuracy
  - exact_match
pipeline_tag: text-generation
---

# md-reheader

A fine-tuned Qwen3-0.6B model that restores correct heading hierarchy
in markdown documents up to 32k tokens.

## Model Description

<!-- Fill in after training -->

## Training

<!-- Fill in after training -->

## Evaluation Results

<!-- Fill in after evaluation -->

## Usage

<!-- Fill in after publishing -->

## Limitations

- Trained primarily on English technical documentation from GitHub
- May underperform on legal, medical, or literary documents
- Assumes headings are already identified (does not detect headings in plain text)
