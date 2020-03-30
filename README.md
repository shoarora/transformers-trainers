# lmtuners

This repo contains trainers for language model pre-training tasks.
Currently, there are two kinds:

1.  `LMTrainer` (normal/causal LM as well as masked LM)
2.  `DiscLMTrainer` (discriminative language modelling task from [ELECTRA paper](https://openreview.net/pdf?id=r1xMH1BtvB))

We've only built small models with this library (fit on one GPU), but the code _theoretically_ generalizes to bigger models.
We don't have the resources to experiment with that, but it should be relatively easy to adapt
the lightning modules to other needs.

## Dependencies

This package is built on top of:

-   huggingface/transformers
    -   model implementations, `*ForMaskedLM`, `*ForTokenClassification`, and optimizers
-   huggingface/tokenizers
    -   their Rust-backed fast tokenizers
-   pytorch-lightning
    -   Abstracts training loops, checkpointing, multi-gpu/distributed learning, other training features.
    -   Theoretically supports TPU, but WIP.
-   pytorch-lamb
    -  LAMB optimizer implementation.
