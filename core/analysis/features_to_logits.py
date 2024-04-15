import torch
from transformer_lens import HookedTransformer

from core.sae import SparseAutoEncoder
from core.config import FeaturesDecoderConfig

@torch.no_grad()
def features_to_logits(sae: SparseAutoEncoder, model: HookedTransformer, cfg: FeaturesDecoderConfig):
    num_ones = int(torch.sum(sae.feature_act_mask).item())

    feature_acts = torch.zeros(num_ones, 24576).to(cfg.device)

    index = 0
    for i in range(len(sae.feature_act_mask)):
        if sae.feature_act_mask[i] == 1:
            feature_acts[index, i] = 1
            index += 1

    feature_acts = torch.unsqueeze(feature_acts, dim=1)
    
    residual = sae.features_decoder(feature_acts)
    
    if model.cfg.normalization_type is not None:
        residual = model.ln_final(residual)  # [batch, pos, d_model]
    logits = model.unembed(residual)  # [batch, pos, d_vocab]

    active_indices = [i for i, val in enumerate(sae.feature_act_mask) if val == 1]
    result_dict = {str(feature_index): logits[idx][0] for idx, feature_index in enumerate(active_indices)}

    return result_dict