from functools import partial
from typing import Any, cast

import pandas as pd
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm
from transformer_lens import HookedTransformer

from core.sae import SparseAutoEncoder
from core.activation.activation_store import ActivationStore
# from core.activation_store_theirs import ActivationStoreTheirs
from core.config import LanguageModelSAEConfig

@torch.no_grad()
def run_evals(
    model: HookedTransformer,
    sae: SparseAutoEncoder,
    activation_store: ActivationStore,
    cfg: LanguageModelSAEConfig,
    n_training_steps: int,
):
    hook_point = cfg.hook_point

    ### Evals
    eval_tokens = activation_store.next_tokens(cfg.store_batch_size)

    # Get Reconstruction Score
    losses_df = recons_loss_batched(
        model,
        sae,
        activation_store,
        cfg,
        n_batches=10,
    )

    recons_score = losses_df["score"].mean()
    ntp_loss = losses_df["loss"].mean()
    recons_loss = losses_df["recons_loss"].mean()
    zero_abl_loss = losses_df["zero_abl_loss"].mean()

    # get cache
    _, cache = model.run_with_cache(
        eval_tokens,
        prepend_bos=False,
        names_filter=[hook_point],
    )

    # get act
    original_act = cache[cfg.hook_point]

    _, (_, aux) = sae.forward(original_act, n_training_steps)
    del cache

    if "cuda" in str(model.cfg.device):
        torch.cuda.empty_cache()

    l2_norm_in = torch.norm(original_act, dim=-1)
    # if cfg.l0_type == 'glu':
    #     l2_norm_out = torch.norm(aux["x_hat_thres"], dim=-1)
    # else:
    # FIXME
    l2_norm_out = torch.norm(aux["x_hat"], dim=-1)
    if cfg.use_ddp:
        dist.reduce(l2_norm_in, dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(l2_norm_out, dst=0, op=dist.ReduceOp.AVG)
    l2_norm_ratio = l2_norm_out / l2_norm_in

    # if cfg.l0_type == 'glu':
    #     pseudo_x_hat = aux["x_hat_thres"] / l2_norm_out.unsqueeze(-1) * l2_norm_in.unsqueeze(-1)
    #     l0 = (aux["feature_acts_thres"] > 0).float().sum(-1)
    # else:
    # FIXME
    pseudo_x_hat = aux["x_hat"] / l2_norm_out.unsqueeze(-1) * l2_norm_in.unsqueeze(-1)
    l0 = (aux["feature_acts"] > 0).float().sum(-1)
    explained_variance = 1 - (pseudo_x_hat - original_act).pow(2).sum(dim=-1) / (original_act - original_act.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)).pow(2).sum(dim=-1)

    # TODO: DDP

    metrics = {
        # l2 norms
        "metrics/l2_norm": l2_norm_out.mean().item(),
        "metrics/l2_ratio": l2_norm_ratio.mean().item(),
        # variance explained
        "metrics/explained_variance": explained_variance.mean().item(),
        "metrics/explained_variance_std": explained_variance.std().item(),
        "metrics/l0": l0.mean().item(),
        # CE Loss
        "metrics/ce_loss_score": recons_score,
        "metrics/ce_loss_without_sae": ntp_loss,
        "metrics/ce_loss_with_sae": recons_loss,
        "metrics/ce_loss_with_ablation": zero_abl_loss,
    }

    if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
        wandb.log(
            metrics,
            step=n_training_steps + 1,
        )

    return metrics

def recons_loss_batched(
    model: HookedTransformer,
    sae: SparseAutoEncoder,
    activation_store: ActivationStore,
    cfg: LanguageModelSAEConfig,
    n_batches: int = 100,
):
    losses = []
    if (not cfg.use_ddp or cfg.rank == 0):
        pbar = tqdm(total=n_batches, desc="Evaluation", smoothing=0.01)
    for _ in range(n_batches):
        batch_tokens = activation_store.next_tokens(cfg.store_batch_size)
        score, loss, recons_loss, zero_abl_loss = get_recons_loss(
            model, sae, cfg, batch_tokens
        )
        if cfg.use_ddp:
            dist.reduce(score, dst=0, op=dist.ReduceOp.AVG)
            dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
            dist.reduce(recons_loss, dst=0, op=dist.ReduceOp.AVG)
            dist.reduce(zero_abl_loss, dst=0, op=dist.ReduceOp.AVG)
        losses.append(
            (
                score.mean().item(),
                loss.mean().item(),
                recons_loss.mean().item(),
                zero_abl_loss.mean().item(),
            )
        )
        if (not cfg.use_ddp or cfg.rank == 0):
            pbar.update(1)

    if (not cfg.use_ddp or cfg.rank == 0):
        pbar.close()

    losses = pd.DataFrame(
        losses, columns=cast(Any, ["score", "loss", "recons_loss", "zero_abl_loss"])
    )

    return losses


@torch.no_grad()
def get_recons_loss(
    model: HookedTransformer,
    sae: SparseAutoEncoder,
    cfg: LanguageModelSAEConfig,
    batch_tokens: torch.Tensor,
):
    batch_tokens = batch_tokens.to(torch.int64)
    hook_point = cfg.hook_point
    loss = model.forward(batch_tokens, return_type="loss")

    def replacement_hook(activations: torch.Tensor, hook: Any):
        _, (_, aux) = sae.forward(activations)
        # if cfg.l0_type == 'glu':
        #     activations = aux["x_hat_thres"].to(activations.dtype)
        # else:
        # FIXME
        activations = aux["x_hat"].to(activations.dtype)
        return activations
    
    recons_loss: torch.Tensor = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(hook_point, partial(replacement_hook))],
    )

    zero_abl_loss: torch.Tensor = model.run_with_hooks(
        batch_tokens, return_type="loss", fwd_hooks=[(hook_point, zero_ablate_hook)]
    )

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    return score, loss, recons_loss, zero_abl_loss


def zero_ablate_hook(activations: torch.Tensor, hook: Any):
    activations = torch.zeros_like(activations)
    return activations
