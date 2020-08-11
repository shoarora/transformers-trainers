import torch
from torch import nn
from transformers import (get_constant_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)


def tie_weights(output_embeddings, input_embeddings):
    """ Tie module weights
    """
    output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())

    if hasattr(output_embeddings, "bias") and output_embeddings.bias is not None:
        output_embeddings.bias.data = torch.nn.functional.pad(
            output_embeddings.bias.data,
            (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
            "constant",
            0,
        )
    if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
        output_embeddings.out_features = input_embeddings.num_embeddings


def get_lr_schedule(optimizer, schedule_type, warmup_steps, total_steps):
    if schedule_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    elif schedule_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
    elif schedule_type == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps
        )
    else:
        raise Exception(f"schedule_type: {schedule_type} not supported")
    return scheduler
