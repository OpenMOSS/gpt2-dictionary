from typing import Dict
import torch
import math
from einops import einsum

from core.config import LanguageModelSAETrainingConfig
from core.utils import compute_geometric_median

class SparseAutoEncoder(torch.nn.Module):
    def __init__(
            self,
            cfg: LanguageModelSAETrainingConfig
    ):
        super(SparseAutoEncoder, self).__init__()

        self.cfg = cfg

        self.encoder = torch.nn.Parameter(torch.empty((cfg.d_model, cfg.d_sae), dtype=cfg.dtype))
        torch.nn.init.kaiming_uniform_(self.encoder)

        self.decoder = torch.nn.Parameter(torch.empty((cfg.d_sae, cfg.d_model), dtype=cfg.dtype))
        torch.nn.init.kaiming_uniform_(self.decoder)
        self.set_decoder_norm_to_unit_norm()

        self.decoder_bias = torch.nn.Parameter(torch.empty((cfg.d_model,), dtype=cfg.dtype))
        torch.nn.init.zeros_(self.decoder_bias)

        self.encoder_bias = torch.nn.Parameter(torch.empty((cfg.d_sae,), dtype=cfg.dtype))
        torch.nn.init.zeros_(self.encoder_bias)

    def forward(self, x: torch.Tensor, dead_neuron_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        if self.cfg.norm_activation:
            # Normalize the activation vectors to have L2 norm equal to sqrt(d_model)
            # x_norms: (batch_size,)
            x_norms = torch.norm(x, 2, dim=-1)

            # x: (batch_size, d_model)
            x = x / torch.clamp(x_norms.unsqueeze(-1), 1e-8) * math.sqrt(self.cfg.d_model)
        
        # hidden_pre: (batch_size, d_sae)
        hidden_pre = einsum(
            x - self.decoder_bias,
            self.encoder,
            "... d_model, d_model d_sae -> ... d_sae",
        ) + self.encoder_bias

        # feature_acts: (batch_size, d_sae)
        feature_acts = torch.clamp(hidden_pre, min=0.0)

        # x_hat: (batch_size, d_model)
        x_hat = einsum(
            feature_acts,
            self.decoder,
            "... d_sae, d_sae d_model -> ... d_model",
        ) + self.decoder_bias

        # Take the sum of the dense dimension in MSE loss
        # l_rec: (batch_size, d_model)
        l_rec = (x_hat - x).pow(2) / (x - x.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()

        # l_l1: (batch_size,)
        l_l1 = torch.norm(feature_acts, p=1, dim=-1)

        l_ghost_resid = torch.tensor(0.0, dtype=self.cfg.dtype, device=self.cfg.device)

        # gate on config and training so evals is not slowed down.
        if (
            self.cfg.use_ghost_grads
            and self.training
            and dead_neuron_mask is not None
            and dead_neuron_mask.sum() > 0
        ):
            # ghost protocol

            # 1.
            residual = x - x_hat
            residual_centred = residual - residual.mean(dim=0, keepdim=True)
            l2_norm_residual = torch.norm(residual, dim=-1)

            # 2.
            feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
            ghost_out = feature_acts_dead_neurons_only @ self.decoder[dead_neuron_mask, :]
            l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
            norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
            ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

            # 3.
            l_ghost_resid = (
                torch.pow((ghost_out - residual.detach().float()), 2)
                / (residual_centred.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
            )
            mse_rescaling_factor = (l_rec / (l_ghost_resid + 1e-6)).detach()
            l_ghost_resid = mse_rescaling_factor * l_ghost_resid

        if self.cfg.norm_activation:
            # Recover the original scale of the activation vectors
            # x_hat: (batch_size, activation_size)
            x_hat = x_hat * x_norms.unsqueeze(-1) / math.sqrt(self.cfg.d_model)

        loss_data = {
            "l_rec": l_rec,
            "l_l1": l_l1,
            "l_ghost_resid": l_ghost_resid,
        }

        aux_data = {
            "feature_acts": feature_acts,
            "x_hat": x_hat,
        }

        return l_rec.mean() + self.cfg.l1_coefficient * l_l1.mean() + l_ghost_resid.mean(), (loss_data, aux_data)
    
    @torch.no_grad()
    def initialize_decoder_bias(self, all_activations: torch.Tensor):
        if self.cfg.decoder_bias_init_method == "geometric_median":
            self.initialize_decoder_bias_with_geometric_median(all_activations)
        elif self.cfg.decoder_bias_init_method == "mean":
            self.initialize_decoder_bias_with_mean(all_activations)
        elif self.cfg.decoder_bias_init_method == "zeros":
            pass
        else:
            raise ValueError(
                f"Unexpected b_dec_init_method: {self.cfg.decoder_bias_init_method}"
            )

    @torch.no_grad()
    def initialize_decoder_bias_with_geometric_median(self, all_activations: torch.Tensor):
        assert self.cfg.geometric_median_max_iter is not None

        previous_decoder_bias = self.decoder_bias.clone()
        out = compute_geometric_median(
            all_activations, max_iter=self.cfg.geometric_median_max_iter
        )

        previous_distances = torch.norm(all_activations - previous_decoder_bias, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with geometric median of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.decoder_bias.data = out

    @torch.no_grad()
    def initialize_decoder_bias_with_mean(self, all_activations: torch.Tensor):
        previous_decoder_bias = self.decoder_bias.clone()
        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_decoder_bias, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing decoder with mean of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.decoder_bias.data = out
    
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.decoder.data /= torch.norm(self.decoder.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_model) shape
        """

        parallel_component = einsum(
            self.decoder.grad,
            self.decoder.data,
            "d_sae d_model, d_sae d_model -> d_sae",
        )

        self.decoder.grad -= einsum(
            parallel_component,
            self.decoder.data,
            "d_sae, d_sae d_model -> d_sae d_model",
        )
    
    @torch.no_grad()
    def compute_thomson_potential(self):
        dist = torch.cdist(self.decoder, self.decoder, p=2).flatten()[1:].view(self.cfg.d_sae - 1, self.cfg.d_sae + 1)[:, :-1]
        mean_thomson_potential = (1 / dist).mean()
        return mean_thomson_potential