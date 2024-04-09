from typing import Dict
import torch
import torch.nn.functional as F 
import math
from einops import einsum
import pdb

from core.config import L0SAEConfig
from core.utils.math import compute_geometric_median
from core.l0layer.glu import L0LayerGLU

class L0SparseAutoEncoder(torch.nn.Module):
    def __init__(
            self,
            cfg: L0SAEConfig
    ):
        super(L0SparseAutoEncoder, self).__init__()

        self.cfg = cfg
        self.device_1 = torch.device('cuda:1') if torch.cuda.device_count() < 5 else torch.device('cuda:5')

        self.encoder = torch.nn.Parameter(torch.empty((cfg.d_model, cfg.d_sae), dtype=cfg.dtype, device=cfg.device))
        torch.nn.init.kaiming_uniform_(self.encoder)

        if cfg.use_glu_encoder:
            # GAU Encoder
            self.encoder_glu = torch.nn.Parameter(torch.empty((cfg.d_model, cfg.d_sae), dtype=cfg.dtype, device=cfg.device))
            torch.nn.init.kaiming_uniform_(self.encoder_glu)

            self.encoder_bias_glu = torch.nn.Parameter(torch.empty((cfg.d_sae,), dtype=cfg.dtype, device=cfg.device))
            torch.nn.init.zeros_(self.encoder_bias_glu)
            # self.hidden_pre_glu = torch.nn.Parameter(torch.empty((cfg.d_sae,), dtype=cfg.dtype, device=cfg.device))
            # torch.nn.init.normal_(self.hidden_pre_glu)


        self.feature_act_mask = torch.nn.Parameter(torch.ones((cfg.d_sae,), dtype=cfg.dtype, device=cfg.device))
        self.feature_act_scale = torch.nn.Parameter(torch.ones((cfg.d_sae,), dtype=cfg.dtype, device=cfg.device))

        self.decoder = torch.nn.Parameter(torch.empty((cfg.d_sae, cfg.d_model), dtype=cfg.dtype, device=cfg.device))
        torch.nn.init.kaiming_uniform_(self.decoder)
        self.set_decoder_norm_to_unit_norm()

        if cfg.use_decoder_bias:
            self.decoder_bias = torch.nn.Parameter(torch.empty((cfg.d_model,), dtype=cfg.dtype, device=cfg.device))
            torch.nn.init.zeros_(self.decoder_bias)

        self.encoder_bias = torch.nn.Parameter(torch.empty((cfg.d_sae,), dtype=cfg.dtype, device=cfg.device))
        torch.nn.init.zeros_(self.encoder_bias)

        self.train_base_parameters()

    def train_base_parameters(self):
        base_parameters = [
            self.encoder,
            self.decoder,
            self.encoder_bias,
        ]
        if self.cfg.use_glu_encoder:
            base_parameters.extend([self.encoder_glu, self.encoder_bias_glu])
            # base_parameters.extend([self.hidden_pre_glu])
        if self.cfg.use_decoder_bias:
            base_parameters.append(self.decoder_bias)
        for p in self.parameters():
            p.requires_grad_(False)
        for p in base_parameters:
            p.requires_grad_(True)

    def train_finetune_for_suppresion_parameters(self):
        finetune_for_suppression_parameters = [
            self.feature_act_scale,
            self.decoder,
        ]
        if self.cfg.use_decoder_bias:
            finetune_for_suppression_parameters.append(self.decoder_bias)
        for p in self.parameters():
            p.requires_grad_(False)
        for p in finetune_for_suppression_parameters:
            p.requires_grad_(True)
        

    def compute_norm_factor(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize the activation vectors to have L2 norm equal to sqrt(d_model)
        if self.cfg.norm_activation == "token-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(x, 2, dim=-1, keepdim=True)
        elif self.cfg.norm_activation == "batch-wise":
            return math.sqrt(self.cfg.d_model) / torch.norm(x, 2, dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        else:
            return torch.tensor(1.0, dtype=self.cfg.dtype, device=self.cfg.device)

    def forward(self, x: torch.Tensor, dead_neuron_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        # norm_factor: (batch_size,)
        # pdb.set_trace()
        norm_factor = self.compute_norm_factor(x)
        if self.cfg.l0_type == 'glu':
            gate_norm_factor = self.compute_norm_factor(self.encoder_glu)

        x = x * norm_factor

        if self.cfg.use_decoder_bias:
            x = x - self.decoder_bias
        
        # hidden_pre: (batch_size, d_sae)
        hidden_pre = einsum(
            x,
            self.encoder,
            "... d_model, d_model d_sae -> ... d_sae",
        ) + self.encoder_bias

        #FIXME
        if self.cfg.l0_type == 'glu':
            # hidden_pre_glu: (batch_size, d_sae)
            hidden_pre_glu = einsum(
                x,
                self.encoder_glu * gate_norm_factor,
                "... d_model, d_model d_sae -> ... d_sae",
            ) + self.encoder_bias_glu
            # hidden_pre_glu = torch.clamp(hidden_pre_glu, 0, 1)
            hidden_pre_glu = F.sigmoid(hidden_pre_glu)
            # hidden_pre_glu = F.threshold(hidden_pre_glu, 1e-2, 0, inplace=False)
            # feature_acts: (batch_size, d_sae)
            feature_acts = self.feature_act_mask * self.feature_act_scale * torch.clamp(hidden_pre, min=0.0) * hidden_pre_glu

            # x_hat: (batch_size, d_model)
            x_hat = einsum(
                feature_acts,
                self.decoder,
                "... d_sae, d_sae d_model -> ... d_model",
            )
            with torch.no_grad():
                feature_acts_thres = self.feature_act_mask * self.feature_act_scale * torch.clamp(hidden_pre, min=0.0) * F.threshold(hidden_pre_glu, 0.5, 0, inplace=False)
                x_hat_thres = einsum(
                    feature_acts_thres,
                    self.decoder,
                    "... d_sae, d_sae d_model -> ... d_model",
                )
                
        else:
            # feature_acts: (batch_size, d_sae)
            feature_acts = self.feature_act_mask * self.feature_act_scale * torch.clamp(hidden_pre, min=0.0)

            # x_hat: (batch_size, d_model)
            x_hat = einsum(
                feature_acts,
                self.decoder,
                "... d_sae, d_sae d_model -> ... d_model",
            )

        # Take the sum of the dense dimension in MSE loss
        # l_rec: (batch_size, d_model)
        l_rec = (x_hat - x).pow(2) / (x - x.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()

        # l_l1: (batch_size,)
        l_l1 = torch.norm(feature_acts, p=self.cfg.lp, dim=-1)

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

        if self.cfg.use_decoder_bias:
            x_hat = x_hat + self.decoder_bias

        # Recover the original scale of the activation vectors
        # x_hat: (batch_size, activation_size)
        x_hat = x_hat / norm_factor

        # l_l0: (batch_size,)
        #FIXME

        loss_data = {
            "l_rec": l_rec,
            "l_l1": l_l1,
            "l_ghost_resid": l_ghost_resid,
        }
        if self.cfg.l0_type == "glu":
            loss_data.update({"l_gate": torch.norm(hidden_pre_glu, p=self.cfg.lp, dim=-1)})

        aux_data = {
            "feature_acts": feature_acts,
            "x_hat": x_hat,
        }
        if self.cfg.l0_type == "glu":
            aux_data.update({"encoder_glu": self.encoder_glu})
            aux_data.update({"feature_acts_thres": feature_acts_thres})
            aux_data.update({"x_hat_thres": x_hat_thres / norm_factor})

        loss = l_rec.mean() + self.cfg.l1_coefficient * l_l1.mean() + l_ghost_resid.mean()
        #FIXME
        return  loss, (loss_data, aux_data)
    
    @torch.no_grad()
    def initialize_decoder_bias(self, all_activations: torch.Tensor):
        if not self.cfg.use_decoder_bias:
            raise ValueError("Decoder bias is not used!")
        norm_factor = self.compute_norm_factor(all_activations)
        all_activations = all_activations * norm_factor
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
        decoder_norm = torch.norm(self.decoder, dim=1, keepdim=True)
        if self.cfg.decoder_exactly_unit_norm:
            self.decoder.data = self.decoder.data / decoder_norm
        else:
            # Set the norm of the decoder to not exceed 1
            self.decoder.data = self.decoder.data / torch.clamp(decoder_norm, min=1.0)
        

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
    
    @torch.no_grad()
    def features_decoder(self, feature_acts):
        x_hat = einsum(
            feature_acts,
            self.decoder,
            "... d_sae, d_sae d_model -> ... d_model",
        )
        return x_hat