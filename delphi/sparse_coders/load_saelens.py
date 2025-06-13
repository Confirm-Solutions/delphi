from functools import partial
from pathlib import Path
from typing import Callable

import torch
from sae_lens import SAE
from torch import Tensor
from transformer_lens import HookedTransformer
from transformers import PreTrainedModel


def sae_dense_latents_saelens(x: Tensor, sae: SAE) -> Tensor:
    """Run SAELens SAE on x, yielding the dense activations."""
    x_in = x.reshape(-1, x.shape[-1])
    acts = sae.encode(x_in)
    # Some SAE.encode methods may return a tuple (acts, ...), so take acts if so
    if isinstance(acts, tuple):
        acts = acts[0]
    return acts.reshape(*x.shape[:-1], -1)


def load_saelens_sparse_coders(
    names: list[str],
    hookpoints: list[str],
    device: str | torch.device,
) -> dict[str, SAE]:
    """
    Load SAELens SAE for specified hookpoint.
    Args:
        name (str): Path to directory containing SAE checkpoint, or HuggingFace repo.
        hookpoint (str): Hookpoint to identify the SAE.
        device (str | torch.device): Device to load the SAE on.
    Returns:
        SAE: Loaded SAE.
    """
    name_paths = [Path(name) for name in names]
    if all(path.exists() for path in name_paths):
        sae_dict = {}
        for hookpoint, path in zip(hookpoints, name_paths):
            sae = SAE.load_from_disk(str(path), device=str(device))
            sae_dict[hookpoint] = sae
        return sae_dict
    else:
        # Assume HuggingFace repo: name=release, hookpoint=sae_id
        sae_dict = {}
        for hookpoint, name in zip(hookpoints, names):
            sae, _, _ = SAE.from_pretrained(name, hookpoint, device=str(device))
            sae_dict[hookpoint] = sae
        return sae_dict


def load_saelens_hooks(
    model: PreTrainedModel | HookedTransformer,
    names: list[str],
    hookpoints: list[str],
    device: str | torch.device | None = None,
) -> dict[str, Callable[[Tensor], Tensor]]:
    """
    Load encode functions for SAELens SAEs on specified hookpoints.
    Args:
        model (PreTrainedModel): The model to load autoencoders for.
        name (str): Path to directory or HuggingFace repo.
        hookpoints (list[str]): List of hookpoints to identify the SAEs.
        device (str | torch.device | None): Device to load the SAEs on.
    Returns:
        dict[str, Callable[[Tensor], Tensor]]: Mapping from resolved hookpoint to encode function.
    """  # noqa: E501
    resolved_device = next(model.parameters()).device
    sae_dict = load_saelens_sparse_coders(names, hookpoints, resolved_device)
    hookpoint_to_sparse_encode: dict[str, Callable[[Tensor], Tensor]] = {}
    for hookpoint, sae in sae_dict.items():
        hookpoint_to_sparse_encode[hookpoint] = partial(
            sae_dense_latents_saelens, sae=sae
        )
    return hookpoint_to_sparse_encode


# Objective
# Evaluate sparse autoencoders using an automated interp pieplein
# from Eleuther - Delphi
# Currently supports the "sparsify" backend, which is likely their own SAE implementation
# We will just get it to work with a different implementation (SAElens) - which we use to train ours
