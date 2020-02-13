# lmtuners

This repo contains trainers for language model pre-training tasks.
Currently, there are two kinds:

1.  `LMTrainer` (normal/causal LM as well as masked LM)
2.  `DiscLMTrainer` (discriminative language modelling task from [ELECTRA paper](https://openreview.net/pdf?id=r1xMH1BtvB))

## Dependencies

This package is built on top of:

-   huggingface/transformers
    -   model implementations, `*ForMaskedLM`, `*ForTokenClassification`, and optimizers
-   huggingface/tokenizers
    -   their Rust-backed fast tokenizers
    -   NOTE: we currently use `tokenizers==0.2.1` directly.  `BertTokenizerFast` was not stable at time of dev.
-   pytorch-lightning
    -   Abstracts training loops, checkpointing, multi-gpu/distributed learning, other training features.
-   pytorch-lamb
    -  LAMB optimizer implementation.
