import os
from importlib.metadata import version
from typing import cast

import torch
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import safetensors.torch as safe

from transformer_lens import HookedTransformer

from tqdm import tqdm
import wandb

from lm_saes.activation.activation_store import ActivationStore
from lm_saes.sae import SparseAutoEncoder
from lm_saes.config import LanguageModelSAEPruningConfig, LanguageModelSAETrainingConfig
from lm_saes.optim import get_scheduler
from lm_saes.evals import run_evals
from lm_saes.utils.misc import print_once

def train_sae(
    model: HookedTransformer,
    sae: SparseAutoEncoder,
    activation_store: ActivationStore,
    cfg: LanguageModelSAETrainingConfig,
):
    total_training_tokens = cfg.total_training_tokens
    total_training_steps = total_training_tokens // cfg.effective_batch_size

    print_once(f"Total Training Tokens: {total_training_tokens}")
    print_once(f"Total Training Steps: {total_training_steps}")
    
    n_training_steps = 0
    n_training_tokens = 0
    log_feature_sparsity = None

    checkpoint_thresholds = []
    if cfg.n_checkpoints > 0:
        checkpoint_thresholds = list(
            range(0, total_training_tokens, total_training_tokens // cfg.n_checkpoints)
        )[1:]

    activation_store.initialize()

    # Initialize the SAE decoder bias if necessary
    # if cfg.use_decoder_bias and (not cfg.use_ddp or cfg.rank == 0):
    #     sae.initialize_decoder_bias(activation_store._store[cfg.hook_point_in])

    if cfg.use_ddp:
        ddp = DDP(sae, device_ids=[cfg.rank], output_device=cfg.sae.device)

    act_freq_scores = torch.zeros(cfg.sae.d_sae, device=cfg.sae.device, dtype=cfg.sae.dtype)
    n_forward_passes_since_fired = torch.zeros(cfg.sae.d_sae, device=cfg.sae.device, dtype=cfg.sae.dtype)
    n_frac_active_tokens = torch.tensor([0], device=cfg.sae.device, dtype=torch.int)

    optimizer = Adam(sae.parameters(), lr=cfg.lr, betas=cfg.betas)

    scheduler = get_scheduler(
        cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps=cfg.lr_warm_up_steps,
        cool_down_steps=cfg.lr_cool_down_steps,
        training_steps=total_training_steps,
        lr_end=cfg.lr_end,
    )

    scheduler.step()

    if not cfg.use_ddp or cfg.rank == 0:
        pbar = tqdm(total=total_training_tokens, desc="Training SAE", smoothing=0.01)
    while n_training_tokens < total_training_tokens:
        sae.train()
        sae.update_l1_coefficient(n_training_steps)
        # Get the next batch of activations
        batch = activation_store.next(batch_size=cfg.train_batch_size)
        assert batch is not None, "Activation store is empty"
        activation_in, activation_out = batch[cfg.sae.hook_point_in], batch[cfg.sae.hook_point_out]

        scheduler.step()
        optimizer.zero_grad()

        ghost_grad_neuron_mask = (
            n_forward_passes_since_fired
            > cfg.dead_feature_window
        ).bool()

        # Forward pass
        (
            loss,
            (
                loss_data,
                aux_data,
            )
        ) = sae.compute_loss(
            activation_in,
            dead_feature_mask=ghost_grad_neuron_mask,
            label=activation_out,
        )
        
        did_fire = (aux_data["feature_acts"] > 0).float().sum(0) > 0
        n_forward_passes_since_fired += 1
        n_forward_passes_since_fired[did_fire] = 0
        if cfg.use_ddp:
            dist.all_reduce(n_forward_passes_since_fired, op=dist.ReduceOp.MIN)

        if cfg.finetuning:
            loss = loss_data['l_rec'].mean()
        loss.backward()

        if cfg.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.clip_grad_norm)
        if cfg.remove_gradient_parallel_to_decoder_directions:
            sae.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()

        if not cfg.sae.sparsity_include_decoder_norm:
            sae.set_decoder_norm_to_fixed_norm(1)
        with torch.no_grad():
            act_freq_scores += (aux_data["feature_acts"].abs() > 0).float().sum(0)
            n_frac_active_tokens += activation_in.size(0)

            n_tokens_current = torch.tensor(activation_in.size(0), device=cfg.sae.device, dtype=torch.int)
            if cfg.use_ddp:
                dist.reduce(n_tokens_current, dst=0)
            n_training_tokens += cast(int, n_tokens_current.item())

            # log and then reset the feature sparsity every feature_sampling_window steps
            if (n_training_steps + 1) % cfg.feature_sampling_window == 0:
                if cfg.use_ddp:
                    dist.reduce(act_freq_scores, dst=0)
                    dist.reduce(n_frac_active_tokens, dst=0)
                if cfg.wandb.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
                    feature_sparsity = act_freq_scores / n_frac_active_tokens
                    log_feature_sparsity = torch.log10(feature_sparsity + 1e-10)
                    wandb_histogram = wandb.Histogram(log_feature_sparsity.detach().cpu().float().numpy())
                    wandb.log(
                        {
                            "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                            "plots/feature_density_line_chart": wandb_histogram,
                            "sparsity/below_1e-5": (feature_sparsity < 1e-5).sum().item(),
                            "sparsity/below_1e-6": (feature_sparsity < 1e-6).sum().item(),
                        },
                        step=n_training_steps + 1,
                    )

                act_freq_scores = torch.zeros(cfg.sae.d_sae, device=cfg.sae.device)
                n_frac_active_tokens = torch.tensor([0], device=cfg.sae.device, dtype=torch.int)

            if (n_training_steps + 1) % cfg.log_frequency == 0:
                # metrics for currents acts
                l0 = (aux_data["feature_acts"] > 0).float().sum(-1).mean()
                l_rec = loss_data["l_rec"].mean()
                l_l1 = loss_data["l_l1"].mean()
                l_ghost_resid = loss_data["l_ghost_resid"].mean()

                if cfg.use_ddp:
                    dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
                    dist.reduce(l0, dst=0, op=dist.ReduceOp.AVG)
                    dist.reduce(l_rec, dst=0, op=dist.ReduceOp.AVG)
                    dist.reduce(l_l1, dst=0, op=dist.ReduceOp.AVG)
                    dist.reduce(l_ghost_resid, dst=0, op=dist.ReduceOp.AVG)

                per_token_l2_loss = (aux_data["reconstructed"] - activation_out).pow(2).sum(dim=-1)
                total_variance = (activation_out - activation_out.mean(0)).pow(2).sum(dim=-1)

                l2_norm_error = per_token_l2_loss.sqrt().mean()
                l2_norm_error_ratio = l2_norm_error / activation_out.norm(p=2, dim=-1).mean()


                if cfg.use_ddp:
                    dist.reduce(l2_norm_error, dst=0, op=dist.ReduceOp.AVG)
                    dist.reduce(l2_norm_error_ratio, dst=0, op=dist.ReduceOp.AVG)

                    if cfg.rank == 0:
                        per_token_l2_loss_list = [torch.zeros_like(per_token_l2_loss) for _ in range(dist.get_world_size())]
                        total_variance_list = [torch.zeros_like(total_variance) for _ in range(dist.get_world_size())]
                    dist.gather(per_token_l2_loss, per_token_l2_loss_list if cfg.rank == 0 else None, dst=0)
                    dist.gather(total_variance, total_variance_list if cfg.rank == 0 else None, dst=0)
                    if cfg.rank == 0:
                        per_token_l2_loss = torch.cat(per_token_l2_loss_list, dim=0)
                        total_variance = torch.cat(total_variance_list, dim=0)

                explained_variance = 1 - per_token_l2_loss / total_variance

                # mean_thomson_potential = sae_module.compute_thomson_potential()

                current_learning_rate = optimizer.param_groups[0]["lr"]

                if cfg.wandb.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
                    wandb.log(
                        {
                            # losses
                            "losses/mse_loss": l_rec.item(),
                            "losses/l1_loss": l_l1.item(),
                            "losses/ghost_grad_loss": l_ghost_resid.item(),
                            "losses/overall_loss": loss.item(),
                            # variance explained
                            "metrics/explained_variance": explained_variance.mean().item(),
                            "metrics/explained_variance_std": explained_variance.std().item(),
                            "metrics/l0": l0.item(),
                            # "metrics/mean_thomson_potential": mean_thomson_potential.item(),
                            "metrics/l2_norm_error": l2_norm_error.item(),
                            "metrics/l2_norm_error_ratio": l2_norm_error_ratio.item(),
                            # norm
                            "metrics/decoder_norm": sae.decoder_norm.item(),
                            "metrics/encoder_norm": sae.encoder_norm.item(),
                            "metrics/decoder_bias_mean": sae.decoder_bias.mean().item() if sae.cfg.use_decoder_bias else 0,
                            "metrics/enocder_bias_mean": sae.encoder_bias.mean().item(),
                            # sparsity
                            "sparsity/l1_coefficient": sae.current_l1_coefficient,
                            "sparsity/mean_passes_since_fired": n_forward_passes_since_fired.mean().item(),
                            "sparsity/dead_features": ghost_grad_neuron_mask.sum().item(),
                            "sparsity/useful_features": sae.decoder.norm(p=2, dim=1).gt(0.99).sum().item(),
                            "details/current_learning_rate": current_learning_rate,
                            "details/n_training_tokens": n_training_tokens,
                        },
                        step=n_training_steps + 1,
                    )

            # record loss frequently, but not all the time.
            if (
                (n_training_steps + 1) % (cfg.eval_frequency) == 0
            ):
                sae.eval()
                run_evals(
                    sae=sae,
                    activation_store=activation_store,
                    model=model,
                    cfg=cfg,
                    n_training_steps=n_training_steps,
                )
                sae.train()

            # Checkpoint if at checkpoint frequency
            if len(checkpoint_thresholds) > 0 and n_training_tokens >= checkpoint_thresholds[0] and (
                not cfg.use_ddp or cfg.rank == 0
            ):
                # Save the model and optimizer state
                path = os.path.join(
                    cfg.exp_result_dir, cfg.exp_name, "checkpoints", f"{n_training_steps}.safetensors"
                )
                if not cfg.sae.sparsity_include_decoder_norm:
                    sae.set_decoder_norm_to_fixed_norm(1)
                sae.save_pretrained(path)

                checkpoint_thresholds.pop(0)

            n_training_steps += 1

            if not cfg.use_ddp or cfg.rank == 0:
                l_rec = loss_data["l_rec"].mean().item()
                l_l1 = loss_data["l_l1"].mean().item()
                pbar.set_description(
                    f"{n_training_steps}| MSE Loss {l_rec:.3f} | L1 {l_l1:.3f}"
                )
                pbar.update(n_tokens_current.item())
    
    if not cfg.use_ddp or cfg.rank == 0:
        pbar.close()

    # Save the final model
    if not cfg.use_ddp or cfg.rank == 0:
        path = os.path.join(
            cfg.exp_result_dir, cfg.exp_name, "checkpoints", "final.safetensors"
        )
        if cfg.sae.sparsity_include_decoder_norm:
            sae.transform_to_unit_decoder_norm()
        else:
            sae.set_decoder_norm_to_fixed_norm(1)
        sae.save_pretrained(path)

@torch.no_grad()
def prune_sae(
    sae: SparseAutoEncoder,
    activation_store: ActivationStore,
    cfg: LanguageModelSAEPruningConfig,
):
    sae.eval()
    n_training_tokens = 0
    act_times = torch.zeros(cfg.sae.d_sae, device=cfg.sae.device, dtype=torch.int)
    max_acts = torch.zeros(cfg.sae.d_sae, device=cfg.sae.device, dtype=cfg.sae.dtype)
    activation_store.initialize()

    if cfg.use_ddp:
        ddp = DDP(sae, device_ids=[cfg.rank], output_device=cfg.sae.device)

    if not cfg.use_ddp or cfg.rank == 0:
        pbar = tqdm(total=cfg.total_training_tokens, desc="Pruning SAE", smoothing=0.01)
    while n_training_tokens < cfg.total_training_tokens:
        # Get the next batch of activations
        batch = activation_store.next(batch_size=cfg.train_batch_size)
        assert batch is not None, "Activation store is empty"
        activation_in, activation_out = batch[cfg.sae.hook_point_in], batch[cfg.sae.hook_point_out]

        feature_acts = sae.encode(activation_in, label=activation_out)

        act_times += (feature_acts > 0).int().sum(0)
        max_acts = torch.max(max_acts, feature_acts.max(0).values)

        n_tokens_current = activation_in.size(0)
        if cfg.use_ddp:
            dist.reduce(n_tokens_current, dst=0)
        n_training_tokens += n_tokens_current

        if not cfg.use_ddp or cfg.rank == 0:
            pbar.update(n_tokens_current)

    if not cfg.use_ddp or cfg.rank == 0:
        pbar.close()

    if cfg.use_ddp:
        dist.reduce(act_times, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(max_acts, dst=0, op=dist.ReduceOp.MAX)

    if not cfg.use_ddp or cfg.rank == 0:
        sae.feature_act_mask.data = ((
            act_times > cfg.dead_feature_threshold * cfg.total_training_tokens
        ) & (max_acts > cfg.dead_feature_max_act_threshold) & (sae.decoder.norm(p=2, dim=1) >= cfg.decoder_norm_threshold)).float()
        sae.feature_act_mask.requires_grad_(False)

        if cfg.wandb.log_to_wandb:
            wandb.log(
                {
                    "sparsity/dead_features": (act_times < cfg.dead_feature_threshold * cfg.total_training_tokens).sum().item(),
                    "sparsity/max_acts_below_threshold": (max_acts < cfg.dead_feature_max_act_threshold).sum().item(),
                    "sparsity/decoder_norm_below_threshold": (sae.decoder.norm(p=2, dim=1) < cfg.decoder_norm_threshold).sum().item(),
                    "sparsity/total_pruned_features": (sae.feature_act_mask == 0).sum().item(),
                },
            )

        print("Dead features:", (act_times < cfg.dead_feature_threshold * cfg.total_training_tokens).sum().item())
        print("Max acts below threshold:", (max_acts < cfg.dead_feature_max_act_threshold).sum().item())
        print("Decoder norm below threshold:", (sae.decoder.norm(p=2, dim=1) < cfg.decoder_norm_threshold).sum().item())
        print("Total pruned features:", (sae.feature_act_mask == 0).sum().item())

        path = os.path.join(
            cfg.exp_result_dir, cfg.exp_name, "checkpoints", "pruned.safetensors"
        )
        sae.save_pretrained(path)
        
    return sae
