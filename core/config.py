import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist

import wandb

import os

from core.utils.misc import print_once

from transformer_lens.loading_from_pretrained import get_official_model_name


@dataclass
class RunnerConfig:
    use_ddp: bool = False
    device: str = "cpu"
    hookedmodel_device_temp: str = "cpu"
    seed: int = 42
    dtype: torch.dtype = torch.float32

    exp_name: str = "test"
    exp_series: str | None = None
    exp_result_dir: str = "results"

    def __post_init__(self):
        # Set rank, world_size, and device if using DDP
        if self.use_ddp:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = f"cuda:{self.rank}"

        if not self.use_ddp or self.rank == 0:
            os.makedirs(self.exp_result_dir, exist_ok=True)
            os.makedirs(os.path.join(self.exp_result_dir, self.exp_name), exist_ok=True)


@dataclass
class LanguageModelConfig(RunnerConfig):
    model_name: str = "gpt2"
    hook_point: str = "blocks.0.hook_mlp_out"
    cache_dir: Optional[str] = None
    d_model: int = 768
    local_files_only: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.model_name = get_official_model_name(self.model_name)

    def save_lm_config(self):
        if os.path.exists(
            os.path.join(self.exp_result_dir, self.exp_name, "lm_config.json")
        ):
            raise ValueError(
                f"Language Model Config for {self.exp_name} already exists."
            )
        json.dump(
            {
                k: v
                for k, v in self.__dict__.items()
                if k in LanguageModelConfig.__dataclass_fields__.keys()
                and k not in RunnerConfig.__dataclass_fields__.keys()
            },
            open(
                os.path.join(self.exp_result_dir, self.exp_name, "lm_config.json"),
                "w",
            ),
            indent=4,
        )

    @staticmethod
    def get_lm_config(exp_name: str, exp_result_dir: str) -> dict[str, Any]:
        with open(os.path.join(exp_result_dir, exp_name, "lm_config.json"), "r") as f:
            return json.load(f)

        


@dataclass
class TextDatasetConfig(RunnerConfig):
    dataset_path: str = "openwebtext"
    cache_dir: Optional[str] = None
    is_dataset_tokenized: bool = False
    is_dataset_on_disk: bool = False
    concat_tokens: bool = True
    context_size: int = 128
    store_batch_size: int = 64


@dataclass
class ActivationStoreConfig(LanguageModelConfig, TextDatasetConfig):
    use_cached_activations: bool = False
    cached_activations_path: Optional[str] = (
        None  # Defaults to "activations/{self.dataset_path.split('/')[-1]}/{self.model_name.replace('/', '_')}_{self.context_size}"
    )

    # Activation Store Parameters
    n_tokens_in_buffer: int = 500_000

    def __post_init__(self):
        super().__post_init__()
        # Autofill cached_activations_path unless the user overrode it
        if self.cached_activations_path is None:
            self.cached_activations_path = f"activations/{self.dataset_path.split('/')[-1]}/{self.model_name.replace('/', '_')}_{self.context_size}"


@dataclass
class WandbConfig(RunnerConfig):
    log_to_wandb: bool = True
    wandb_project: str = "gpt2-sae-training"
    run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.run_name is None:
            self.run_name = self.exp_name


@dataclass
class SAEConfig(RunnerConfig):
    """
    Configuration for training or running a sparse autoencoder.
    """

    from_pretrained_path: Optional[str] = None
    strict_loading: bool = True

    use_decoder_bias: bool = True
    decoder_bias_init_method: str = "geometric_median"
    geometric_median_max_iter: Optional[int] = (
        1000  # The maximum number of iterations for the geometric median algorithm. Required if decoder_bias_init_method is geometric_median
    )
    expansion_factor: int = 32
    d_model: int = 768
    d_sae: Optional[int] = (
        None  # The dimension of the SAE, i.e. the number of dictionary components (or features). If None, it will be set to d_model * expansion_factor
    )
    norm_activation: str = "token-wise"  # none, token-wise, batch-wise
    decoder_exactly_unit_norm: bool = True
    use_glu_encoder: bool = False

    l1_coefficient: float = 0.00008
    lp: int = 1

    use_ghost_grads: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.d_sae is None:
            self.d_sae = self.d_model * self.expansion_factor

    def save_hyperparameters(self):
        if os.path.exists(
            os.path.join(self.exp_result_dir, self.exp_name, "hyperparams.json")
        ):
            raise ValueError(
                f"Hyperparams for {self.exp_name} already exists."
            )
        
        # Save hyperparameters (only configs from SAEConfig, excluding derived classes)
        with open(
            os.path.join(
                self.exp_result_dir, self.exp_name, "hyperparams.json"
            ),
            "w",
        ) as f:
            json.dump(
                {
                    k: v
                    for k, v in self.__dict__.items()
                    if k in SAEConfig.__dataclass_fields__.keys()
                    and k not in RunnerConfig.__dataclass_fields__.keys()
                },
                f,
                indent=4,
            )

    @staticmethod
    def get_hyperparameters(
        exp_name: str, exp_result_dir: str, ckpt_name: str, strict_loading: bool = True
    ) -> dict[str, Any]:
        with open(os.path.join(exp_result_dir, exp_name, "hyperparams.json"), "r") as f:
            hyperparams = json.load(f)
        hyperparams["from_pretrained_path"] = os.path.join(
            exp_result_dir, exp_name, "checkpoints", ckpt_name
        )
        hyperparams["strict_loading"] = strict_loading
        return hyperparams

@dataclass
class OpenAIConfig:
    openai_api_key: str
    openai_base_url: str

@dataclass
class AutoInterpConfig(SAEConfig, LanguageModelConfig, OpenAIConfig):
    num_sample: int = 10
    p: float = 0.7
    num_left_token: int = 10
    num_right_token: int = 5


@dataclass
class LanguageModelSAEConfig(SAEConfig, WandbConfig, ActivationStoreConfig):
    pass


@dataclass
class LanguageModelSAETrainingConfig(LanguageModelSAEConfig):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    # Training Parameters
    total_training_tokens: int = 300_000_000
    lr: float = 0.0004
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler_name: str = (
        "constantwithwarmup"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup, exponentialwarmup
    )
    lr_end: Optional[float] = 1 / 32
    lr_warm_up_steps: int = 5000
    lr_cool_down_steps: int = 10000
    train_batch_size: int = 4096

    finetuning: bool = False

    # Resampling protocol args
    feature_sampling_window: int = 1000
    dead_feature_window: int = 5000  # unless this window is larger feature sampling,

    dead_feature_threshold: float = 1e-6

    # Evaluation
    eval_frequency: int = 1000

    # Misc
    log_frequency: int = 10

    n_checkpoints: int = 10

    def __post_init__(self):
        super().__post_init__()

        if not self.use_ddp or self.rank == 0:
            if os.path.exists(
                os.path.join(self.exp_result_dir, self.exp_name, "checkpoints")
            ):
                raise ValueError(
                    f"Ckeckpoints for experiment {self.exp_name} already exist. Consider changing the experiment name."
                )
            os.makedirs(os.path.join(self.exp_result_dir, self.exp_name, "checkpoints"))

        if self.decoder_bias_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.decoder_bias_init_method}"
            )
        if self.decoder_bias_init_method == "zeros":
            print_once(
                "Warning: We are initializing b_dec to zeros. This is probably not what you want."
            )

        self.effective_batch_size = (
            self.train_batch_size * self.world_size
            if self.use_ddp
            else self.train_batch_size
        )
        print_once(f"Effective batch size: {self.effective_batch_size}")

        total_training_steps = self.total_training_tokens // self.effective_batch_size
        print_once(f"Total training steps: {total_training_steps}")

        if self.use_ghost_grads:
            print_once("Using Ghost Grads.")


@dataclass
class LanguageModelSAEPruningConfig(LanguageModelSAEConfig):
    """
    Configuration for pruning a sparse autoencoder on a language model.
    """

    total_training_tokens: int = 10_000_000
    train_batch_size: int = 4096

    dead_feature_threshold: float = 1e-6
    dead_feature_max_act_threshold: float = 1.0
    decoder_norm_threshold: float = 0.99


@dataclass
class ActivationGenerationConfig(LanguageModelConfig, TextDatasetConfig):
    hook_points: list[str] = field(default_factory=list)

    activation_save_path: Optional[str] = (
        None  # Defaults to "activations/{dataset}/{model}_{context_size}"
    )

    total_generating_tokens: int = 300_000_000
    chunk_size: int = int(0.5 * 2**30)  # 0.5 GB

    def __post_init__(self):
        super().__post_init__()

        if self.activation_save_path is None:
            self.activation_save_path = f"activations/{self.dataset_path.split('/')[-1]}/{self.model_name.replace('/', '_')}_{self.context_size}"
        os.makedirs(self.activation_save_path, exist_ok=True)

@dataclass
class MongoConfig:
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "mechinterp"

@dataclass
class LanguageModelSAEAnalysisConfig(SAEConfig, ActivationStoreConfig, MongoConfig):
    """
    Configuration for analyzing a sparse autoencoder on a language model.
    """

    total_analyzing_tokens: int = 300_000_000
    enable_sampling: bool = (
        False  # If True, we will sample the activations based on weights. Otherwise, top n_samples activations will be used.
    )
    sample_weight_exponent: float = 2.0
    subsample: Dict[str, Dict[str, Any]] = field(default_factory=lambda: { "top_activations": {"proportion": 1.0, "n_samples": 10} })


@dataclass
class FeaturesDecoderConfig(SAEConfig, LanguageModelConfig, MongoConfig):
    top: int = 10

@dataclass
class LanguageModelSAEFinetuningConfig(LanguageModelSAEConfig):
    """
    Configuration for training a sparse autoencoder on a language model.
    """
    # Finetuning Parameters
    ft_loss_type: str = None
    ft_kl_coefficient: float = 1.

    # Training Parameters
    total_training_tokens: int = 300_000_000
    lr: float = 1e-5
    use_ghost_grads = False

    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler_name: str = (
        "constantwithwarmup"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup, exponentialwarmup
    )
    lr_end: Optional[float] = 1 / 32
    lr_warm_up_steps: int = 5000
    num_cycles: int = 5
    lr_cool_down_steps: int = 10000
    train_batch_size: int = 4096

    # Resampling protocol args
    feature_sampling_window: int = 1000
    # dead_feature_window: int = 5000  # unless this window is larger feature sampling,

    dead_feature_threshold: float = 1e-6    

    # Evaluation
    eval_frequency: int = 1000

    # Misc
    log_frequency: int = 10

    n_checkpoints: int = 10
    def __post_init__(self):
        super().__post_init__()

        if not self.use_ddp or self.rank == 0:
            os.makedirs(
                os.path.join(self.exp_result_dir, self.exp_name, "analysis"),
                exist_ok=True,
            )
            if os.path.exists(
                os.path.join(
                    self.exp_result_dir, self.exp_name, "analysis", self.analysis_name
                )
            ):
                raise ValueError(
                    f"Analysis {self.analysis_name} for experiment {self.exp_name} already exists. Consider changing the experiment name or the analysis name."
                )
            os.makedirs(
                os.path.join(
                    self.exp_result_dir, self.exp_name, "analysis", self.analysis_name
                )
            )
            if os.path.exists(os.path.join(self.exp_result_dir, self.exp_name, "checkpoints")):
                raise ValueError(f"Checkpoints for experiment {self.exp_name} already exist. Consider changing the experiment name.")
            os.makedirs(os.path.join(self.exp_result_dir, self.exp_name, "checkpoints"))

        if self.decoder_bias_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.decoder_bias_init_method}"
            )
        if self.decoder_bias_init_method == "zeros":
            print_once(
                "Warning: We are initializing b_dec to zeros. This is probably not what you want."
            )

        self.effective_batch_size = self.train_batch_size * self.world_size if self.use_ddp else self.train_batch_size
        print_once(f"Effective batch size: {self.effective_batch_size}")

        total_training_steps = self.total_training_tokens // self.effective_batch_size
        print_once(f"Total training steps: {total_training_steps}")

        if self.use_ghost_grads:
            print_once("Using Ghost Grads.")

@dataclass
class L0SAEConfig(SAEConfig):
    l0_type:str = "glu"    # Could be "glu", "lp", "kl"
    l0_beta:float = 0.01
    n_sample_total:int = 1000
    n_sample_group:int = 10
    lp_p:float = 0.1
    data_parallel:bool = False
    glu_method:str = 'none'
    glu_grad_clip:float = 2e-2

    def __post_init__(self):
        super().__post_init__()
        if self.d_sae is None:
            self.d_sae = self.d_model * self.expansion_factor

@dataclass
class LanguageModelL0SAEConfig(L0SAEConfig, WandbConfig, ActivationStoreConfig):
    pass

@dataclass
class LanguageModelL0SAETrainingConfig(LanguageModelL0SAEConfig):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    # Training Parameters
    total_training_tokens: int = 300_000_000
    lr: float = 0.0004
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler_name: str = (
        "constantwithwarmup"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup, exponentialwarmup
    )
    lr_end: Optional[float] = 1 / 32
    lr_warm_up_steps: int = 5000
    lr_cool_down_steps: int = 10000
    train_batch_size: int = 4096

    # Resampling protocol args
    feature_sampling_window: int = 1000
    dead_feature_window: int = 5000  # unless this window is larger feature sampling,

    dead_feature_threshold: float = 1e-6

    # Evaluation
    eval_frequency: int = 1000

    # Misc
    log_frequency: int = 10

    n_checkpoints: int = 10

    def __post_init__(self):
        super().__post_init__()

        if not self.use_ddp or self.rank == 0:
            if os.path.exists(
                os.path.join(self.exp_result_dir, self.exp_name, "checkpoints")
            ):
                raise ValueError(
                    f"Checkpoints for experiment {self.exp_name} already exist. Consider changing the experiment name."
                )
            os.makedirs(os.path.join(self.exp_result_dir, self.exp_name, "checkpoints"))

        if self.decoder_bias_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.decoder_bias_init_method}"
            )
        if self.decoder_bias_init_method == "zeros":
            print_once(
                "Warning: We are initializing b_dec to zeros. This is probably not what you want."
            )

        self.effective_batch_size = (
            self.train_batch_size * self.world_size
            if self.use_ddp
            else self.train_batch_size
        )
        print_once(f"Effective batch size: {self.effective_batch_size}")

        total_training_steps = self.total_training_tokens // self.effective_batch_size
        print_once(f"Total training steps: {total_training_steps}")

        if self.use_ghost_grads:
            print_once("Using Ghost Grads.")