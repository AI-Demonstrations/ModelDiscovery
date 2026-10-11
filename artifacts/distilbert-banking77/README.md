---
license: apache-2.0
base_model: distilbert-base-uncased
datasets:
- PolyAI/banking77
language:
- en
pipeline_tag: text-classification
tags:
- intent-classification
- course-demo
---
# banking77 intent router (course demo)

## Intended use
Routes English customer-support messages from a retail bank into one of 77 intents.
Out of scope: other languages, other domains, messages that fit none of the 77 intents
(the model will still pick one: use the confidence threshold and fallback queue).

## Training data
banking77 train split (8,993 messages after a 10% validation hold-out), from huggingface.co/datasets/mteb/banking77.
Original license: CC-BY-4.0 (PolyAI).

## Evaluation (frozen test split, fingerprint e62ee22558ec)
- Macro-F1: 0.897 (95% bootstrap CI 0.885-0.906)
- p95 latency, single message on CPU: 13 ms
- Weakest intents by recall: virtual_card_not_working, why_verify_identity, get_disposable_virtual_card, balance_not_updated_after_bank_transfer, declined_transfer

## Decision record
Candidates: A: TF-IDF + LR, B: Zero-shot NLI, C: MiniLM + LR, D: Fine-tuned DistilBERT; hosted APIs excluded (data may not leave the environment).
Weights: {'quality': 0.4, 'latency': 0.15, 'cost': 0.2, 'data_control': 0.15, 'ops': 0.1}. Winner under these weights: C: MiniLM + LR.
Shipped: D (fine-tuned distilbert-base-uncased). If that differs from the winner, record why here: [reason].

## Limitations
Single-domain, English-only, synthetic-feeling phrasing; no out-of-scope class; confidence is not calibrated.

## How to load
`pipeline("text-classification", model=<repo>, revision=<commit sha>, trust_remote_code=False)`
