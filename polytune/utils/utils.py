import torch
from toch import nn


def tie_weights(self, output_embeddings, input_embeddings):
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
