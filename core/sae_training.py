import os

import torch
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
import pdb

from transformer_lens import HookedTransformer

from tqdm import tqdm
import wandb

from core.activation.activation_store import ActivationStore
from core.sae import SparseAutoEncoder
from core.config import LanguageModelSAEPruningConfig, LanguageModelSAETrainingConfig
from core.optim import get_scheduler
from core.evals import run_evals
from core.utils.misc import print_once

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
    if cfg.use_decoder_bias and (not cfg.use_ddp or cfg.rank == 0):
        sae.initialize_decoder_bias(activation_store._store["activation"])

    sae_module = sae
    if cfg.use_ddp:
        sae = DDP(sae, device_ids=[cfg.rank], output_device=cfg.device)
        sae_module: SparseAutoEncoder = sae.module

    assert cfg.d_sae is not None
    act_freq_scores = torch.zeros(cfg.d_sae, device=cfg.device, dtype=cfg.dtype)
    n_forward_passes_since_fired = torch.zeros(cfg.d_sae, device=cfg.device, dtype=cfg.dtype)
    n_frac_active_tokens = torch.tensor([0], device=cfg.device, dtype=torch.int)

    optimizer = Adam(sae.parameters(), lr=cfg.lr, betas=cfg.betas)
    if cfg.sae_from_pretrained_path is not None:
        checkpoint = torch.load(cfg.sae_from_pretrained_path, map_location=cfg.device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

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
        pbar = tqdm(total=total_training_tokens, desc="Training SAE", smoothing=0.01, position=0)
    while n_training_tokens < total_training_tokens:
        if n_training_steps == total_training_steps - cfg.lr_cool_down_steps:
            sae.cfg.use_ghost_grads = False
        sae.train()
        # Get the next batch of activations
        if cfg.l0_type == 'kl':
            tokens = activation_store.next_tokens(cfg.store_batch_size)
            # pdb.set_trace()
            _, cache = model.run_with_cache(tokens, names_filter=[cfg.hook_point])
            batch = rearrange(cache[cfg.hook_point].to(dtype=cfg.dtype, device=cfg.device), "b l d -> (b l) d")
        else:
            batch_dict = activation_store.next(batch_size=cfg.train_batch_size)
            batch = batch_dict["activation"]

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
        ) = sae.forward(
            batch,
            ghost_grad_neuron_mask,
        )
        
        if cfg.glu_method == 'sig':
            did_fire = (aux_data["feature_acts_thres"] > 0).float().sum(0) > 0
        else:
            did_fire = (aux_data["feature_acts"] > 0).float().sum(0) > 0
        n_forward_passes_since_fired += 1
        n_forward_passes_since_fired[did_fire] = 0
        if cfg.use_ddp:
            dist.all_reduce(n_forward_passes_since_fired, op=dist.ReduceOp.MIN)

        if cfg.finetuning:
            loss = loss_data['l_rec'].mean()
        if cfg.l0_type == 'glu':
            # beta_factor = min((n_training_steps + 1) / cfg.lr_warm_up_steps, 1)
            beta_factor = 1
            l_l1 = loss_data['l_l1'].mean()
            loss = loss_data['l_rec'].mean() + beta_factor * cfg.l0_beta * l_l1 + loss_data['l_ghost_resid'].mean()
        elif cfg.l0_type == 'kl':
            sae.eval()
            with torch.no_grad():
                batch_tokens = tokens.to(torch.int64)
                hook_point = cfg.hook_point
                def hook_func(activations: torch.Tensor, hook: any):
                    return activations
                x_logits = model.run_with_hooks(
                    batch_tokens,
                    return_type="logits",
                    fwd_hooks=[(hook_point, hook_func)]
                )
                def hook_func_real(activations: torch.Tensor, hook: any):
                    return rearrange(aux_data['x_hat'],'(b l) d -> b l d', b=16, l=256)
                # pdb.set_trace()
                x_hat_logits = model.run_with_hooks(
                                batch_tokens,
                                return_type="logits",
                                fwd_hooks=[(hook_point, hook_func_real)],
                )
            sae.train()
            input_log = F.log_softmax(x_logits, dim=1)
            output = F.softmax(x_hat_logits, dim=1)
            output_log = F.log_softmax(x_hat_logits, dim=1)
            l_l0 = (output * (output_log - input_log)).sum() / input_log.size(0) 
            #FIXME
            loss = loss_data['l_rec'].mean() + cfg.l0_beta * l_l0 + cfg.l1_coefficient * loss_data['l_l1'].mean() + loss_data['l_ghost_resid'].mean()
        else:
            l_l1 = loss_data['l_l1'].mean()
            loss = loss_data['l_rec'].mean() + cfg.l0_beta * l_l1 + loss_data['l_ghost_resid'].mean()
        loss.backward()
        # torch.cuda.empty_cache()
        sae_module.remove_gradient_parallel_to_decoder_directions()
        if cfg.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=cfg.grad_clip) 
        optimizer.step()

        sae_module.set_decoder_norm_to_unit_norm()
        with torch.no_grad():
            if cfg.glu_method == 'sig':
                act_freq_scores += (aux_data["feature_acts_thres"].abs() > 0).float().sum(0)
            else:
                act_freq_scores += (aux_data["feature_acts"].abs() > 0).float().sum(0)
            n_frac_active_tokens += batch.size(0)

            n_tokens_current = torch.tensor(batch.size(0), device=cfg.device, dtype=torch.int)
            if cfg.use_ddp:
                dist.reduce(n_tokens_current, dst=0)
            n_training_tokens += n_tokens_current.item()

            # log and then reset the feature sparsity every feature_sampling_window steps
            if (n_training_steps + 1) % cfg.feature_sampling_window == 0:
                if cfg.use_ddp:
                    dist.reduce(act_freq_scores, dst=0)
                    dist.reduce(n_frac_active_tokens, dst=0)
                if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
                    feature_sparsity = act_freq_scores / n_frac_active_tokens
                    log_feature_sparsity = torch.log10(feature_sparsity + 1e-10)
                    wandb_histogram = wandb.Histogram(log_feature_sparsity.detach().cpu().numpy())
                    wandb.log(
                        {
                            "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                            "plots/feature_density_line_chart": wandb_histogram,
                            "sparsity/below_1e-5": (feature_sparsity < 1e-5).sum().item(),
                            "sparsity/below_1e-6": (feature_sparsity < 1e-6).sum().item(),
                        },
                        step=n_training_steps + 1,
                    )

                act_freq_scores = torch.zeros(cfg.d_sae, device=cfg.device)
                n_frac_active_tokens = torch.tensor([0], device=cfg.device, dtype=torch.int)

            if ((n_training_steps + 1) % cfg.log_frequency == 0):
                # metrics for currents acts
                if cfg.glu_method == 'sig':
                    l0 = (aux_data["feature_acts_thres"] > 0).float().sum(-1).mean()
                else:
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

                if cfg.glu_method == 'sig':
                    per_token_l2_loss = (aux_data["x_hat_thres"] - batch).pow(2).sum(dim=-1)
                else:
                    per_token_l2_loss = (aux_data["x_hat"] - batch).pow(2).sum(dim=-1)
                total_variance = (batch - batch.mean(0)).pow(2).sum(dim=-1)

                l2_norm_error = per_token_l2_loss.sqrt().mean()
                l2_norm_error_ratio = l2_norm_error / batch.norm(p=2, dim=-1).mean()


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

                if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
                    log_dict = {
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
                            # sparsity
                            "sparsity/mean_passes_since_fired": n_forward_passes_since_fired.mean().item(),
                            "sparsity/dead_features": ghost_grad_neuron_mask.sum().item(),
                            "sparsity/useful_features": sae_module.decoder.norm(p=2, dim=1).gt(0.99).sum().item(),
                            "sparsity/encoder_norm": sae_module.encoder.norm(p=2, dim=1).mean().item(),
                            
                            "details/current_learning_rate": current_learning_rate,
                            "details/n_training_tokens": n_training_tokens,
                        }
                    if cfg.use_glu_encoder:
                        log_dict.update({"sparsity/encoder_glu_norm": aux_data['encoder_glu'].norm(p=2, dim=1).mean().item()})
                    wandb.log(
                        log_dict,
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
                    cfg.exp_result_dir, cfg.exp_name, "checkpoints", f"{n_training_steps}.pt"
                )
                sae_module.set_decoder_norm_to_unit_norm()
                torch.save(
                    {
                        "sae": sae_module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "n_training_steps": n_training_steps,
                        "n_training_tokens": n_training_tokens,
                    },
                    path,
                )

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
            cfg.exp_result_dir, cfg.exp_name, "checkpoints", "final.pt"
        )
        sae_module.set_decoder_norm_to_unit_norm()
        torch.save(
            {
                "sae": sae_module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "n_training_steps": n_training_steps,
                "n_training_tokens": n_training_tokens,
            },
            path,
        )

@torch.no_grad()
def prune_sae(
    sae: SparseAutoEncoder,
    activation_store: ActivationStore,
    cfg: LanguageModelSAEPruningConfig,
):
    sae.eval()
    n_training_tokens = 0
    act_times = torch.zeros(cfg.d_sae, device=cfg.device, dtype=torch.int)
    max_acts = torch.zeros(cfg.d_sae, device=cfg.device, dtype=cfg.dtype)
    activation_store.initialize()

    sae_module = sae
    if cfg.use_ddp:
        sae = DDP(sae, device_ids=[cfg.rank], output_device=cfg.device)
        sae_module: SparseAutoEncoder = sae.module

    if not cfg.use_ddp or cfg.rank == 0:
        pbar = tqdm(total=cfg.total_training_tokens, desc="Pruning SAE", smoothing=0.01)
    while n_training_tokens < cfg.total_training_tokens:
        # Get the next batch of activations
        batch = activation_store.next(batch_size=cfg.train_batch_size)["activation"]

        # Forward pass
        (
            _,
            (
                _,
                aux_data,
            )
        ) = sae.forward(batch)

        act_times += (aux_data["feature_acts"] > 0).int().sum(0)
        max_acts = torch.max(max_acts, aux_data["feature_acts"].max(0).values)

        n_tokens_current = batch.size(0)
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
        sae_module.feature_act_mask.data = ((
            act_times > cfg.dead_feature_threshold * cfg.total_training_tokens
        ) & (max_acts > cfg.dead_feature_max_act_threshold) & (sae_module.decoder.norm(p=2, dim=1) >= cfg.decoder_norm_threshold)).float()
        sae_module.feature_act_mask.requires_grad_(False)

        if cfg.log_to_wandb:
            wandb.log(
                {
                    "sparsity/dead_features": (act_times < cfg.dead_feature_threshold * cfg.total_training_tokens).sum().item(),
                    "sparsity/max_acts_below_threshold": (max_acts < cfg.dead_feature_max_act_threshold).sum().item(),
                    "sparsity/decoder_norm_below_threshold": (sae_module.decoder.norm(p=2, dim=1) < cfg.decoder_norm_threshold).sum().item(),
                    "sparsity/total_pruned_features": (sae_module.feature_act_mask == 0).sum().item(),
                },
            )

        print("Dead features:", (act_times < cfg.dead_feature_threshold * cfg.total_training_tokens).sum().item())
        print("Max acts below threshold:", (max_acts < cfg.dead_feature_max_act_threshold).sum().item())
        print("Decoder norm below threshold:", (sae_module.decoder.norm(p=2, dim=1) < cfg.decoder_norm_threshold).sum().item())
        print("Total pruned features:", (sae_module.feature_act_mask == 0).sum().item())

        path = os.path.join(
            cfg.exp_result_dir, cfg.exp_name, "checkpoints", "pruned.pt"
        )
        torch.save(
            {
                "sae": sae_module.state_dict(),
                "n_training_tokens": n_training_tokens,
            },
            path,
        )
        
    return sae_module