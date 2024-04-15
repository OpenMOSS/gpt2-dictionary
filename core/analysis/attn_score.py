import os
import regex as re

from tqdm import tqdm

import torch

from einops import repeat, rearrange

from datasets import Dataset

from transformer_lens import HookedTransformer

from core.sae import SparseAutoEncoder
from core.config import LanguageModelConfig

@torch.no_grad()
def compute_attention_score(
    model: HookedTransformer,
    lm_cfg: LanguageModelConfig,
    sae1: SparseAutoEncoder,
    sae2: SparseAutoEncoder,
):
    """
    Compute the attention score between two dictionaries, with respect to an specific attention layer.
    """
    hook_point = lm_cfg.hook_point
    result = re.match(r"^blocks\.(\d+)\.hook_attn_out", hook_point)
    assert result is not None, f"Invalid hook point: {hook_point}. Must be of the form 'blocks.<layer>.hook_attn_out'."
    layer = int(result.groups()[0])
    assert 0 <= layer < model.cfg.n_layers, f"Invalid layer: {layer}. Must be in the range [0, {model.cfg.n_layers})."

    # Get the attention weights of QK circuit
    QK = model.blocks[layer].attn.QK
    assert QK is not None, f"Attention weights of layer {layer} not found."

    alive_feature_indices1 = sae1.feature_act_mask.nonzero(as_tuple=True)[0]
    alive_feature_indices2 = sae2.feature_act_mask.nonzero(as_tuple=True)[0]

    # Compute the attention score
    attn_score = (sae1.decoder - sae1.decoder.mean(-1, keepdim=True))[alive_feature_indices1] @ QK @ (sae2.decoder - sae2.decoder.mean(-1, keepdim=True))[alive_feature_indices2].transpose(-2, -1)

    return attn_score, alive_feature_indices1, alive_feature_indices2