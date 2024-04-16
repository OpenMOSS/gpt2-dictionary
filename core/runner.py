from typing import Any, cast
import os

import wandb

import torch

from transformers import AutoModelForCausalLM

from transformer_lens import HookedTransformer

from core.config import ActivationGenerationConfig, LanguageModelSAEAnalysisConfig, LanguageModelSAETrainingConfig, LanguageModelSAEConfig, LanguageModelSAEPruningConfig, FeaturesDecoderConfig
# from core.database import MongoClient
from core.evals import run_evals
from core.sae import SparseAutoEncoder
from core.activation.activation_dataset import make_activation_dataset
from core.activation.activation_store import ActivationStore
from core.sae_training import prune_sae, train_sae
from core.analysis.sample_feature_activations import sample_feature_activations
from core.analysis.features_to_logits import features_to_logits

def language_model_sae_runner(cfg: LanguageModelSAETrainingConfig):
    if not cfg.use_ddp or cfg.rank == 0:
        cfg.save_hyperparameters()
        cfg.save_lm_config()
    sae = SparseAutoEncoder(cfg=cfg)
    if cfg.from_pretrained_path is not None:
        sae.load_state_dict(torch.load(cfg.from_pretrained_path, map_location=cfg.device)["sae"], strict=cfg.strict_loading)

    if cfg.finetuning:
        # Fine-tune SAE with frozen encoder weights and bias
        sae.train_finetune_for_suppresion_parameters()

    hf_model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir=cfg.cache_dir, local_files_only=cfg.local_files_only)
    model = HookedTransformer.from_pretrained('gpt2', device=cfg.device, cache_dir=cfg.cache_dir, hf_model=hf_model)
    model.eval()
    activation_store = ActivationStore.from_config(model=model, cfg=cfg)
        
    if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
        wandb_run = wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name, entity=cfg.wandb_entity)
        with open(os.path.join(cfg.exp_result_dir, cfg.exp_name, "train_wandb_id.txt"), "w") as f:
            f.write(wandb_run.id)
        wandb.watch(sae, log="all")

    # train SAE
    sae = train_sae(
        model,
        sae,
        activation_store,
        cfg,
    )

    if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
        wandb.finish()

    return sae

def language_model_sae_prune_runner(cfg: LanguageModelSAEPruningConfig):
    sae = SparseAutoEncoder(cfg=cfg)
    if cfg.from_pretrained_path is not None:
        sae.load_state_dict(torch.load(cfg.from_pretrained_path, map_location=cfg.device)["sae"], strict=cfg.strict_loading)
    hf_model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir=cfg.cache_dir, local_files_only=cfg.local_files_only)
    model = HookedTransformer.from_pretrained('gpt2', device=cfg.device, cache_dir=cfg.cache_dir, hf_model=hf_model)
    model.eval()
    activation_store = ActivationStore.from_config(model=model, cfg=cfg)
    if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
        wandb_run = wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name, entity=cfg.wandb_entity)
        with open(os.path.join(cfg.exp_result_dir, cfg.exp_name, "prune_wandb_id.txt"), "w") as f:
            f.write(wandb_run.id)

    sae = prune_sae(
        sae,
        activation_store,
        cfg,
    )

    result = run_evals(
        model,
        sae,
        activation_store,
        cfg,
        0
    )

    # Print results in tabular format
    if not cfg.use_ddp or cfg.rank == 0:
        for key, value in result.items():
            print(f"{key}: {value}")

    if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
        wandb.finish()

def language_model_sae_eval_runner(cfg: LanguageModelSAEConfig):
    sae = SparseAutoEncoder(cfg=cfg)
    if cfg.from_pretrained_path is not None:
        sae.load_state_dict(torch.load(cfg.from_pretrained_path, map_location=cfg.device)["sae"], strict=cfg.strict_loading)
    hf_model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir=cfg.cache_dir, local_files_only=cfg.local_files_only)
    model = HookedTransformer.from_pretrained('gpt2', device=cfg.device, cache_dir=cfg.cache_dir, hf_model=hf_model)
    model.eval()
    activation_store = ActivationStore.from_config(model=model, cfg=cfg)
        
    if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
        wandb_run = wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name, entity=cfg.wandb_entity)
        with open(os.path.join(cfg.exp_result_dir, cfg.exp_name, "eval_wandb_id.txt"), "w") as f:
            f.write(wandb_run.id)

    result = run_evals(
        model,
        sae,
        activation_store,
        cfg,
        0
    )

    # Print results in tabular format
    if not cfg.use_ddp or cfg.rank == 0:
        for key, value in result.items():
            print(f"{key}: {value}")

    if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
        wandb.finish()

    return sae

def activation_generation_runner(cfg: ActivationGenerationConfig):
    model = HookedTransformer.from_pretrained('gpt2', device=cfg.device, cache_dir=cfg.cache_dir)
    model.eval()
    
    make_activation_dataset(model, cfg)

def sample_feature_activations_runner(cfg: LanguageModelSAEAnalysisConfig):
    sae = SparseAutoEncoder(cfg=cfg)
    if cfg.from_pretrained_path is not None:
        sae.load_state_dict(torch.load(cfg.from_pretrained_path, map_location=cfg.device)["sae"], strict=cfg.strict_loading)

    hf_model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir=cfg.cache_dir, local_files_only=cfg.local_files_only)
    model = HookedTransformer.from_pretrained('gpt2', device=cfg.device, cache_dir=cfg.cache_dir, hf_model=hf_model)
    model.eval()

    activation_store = ActivationStore.from_config(model=model, cfg=cfg)
    result = sample_feature_activations(sae, model, activation_store, cfg)

    client = MongoClient(cfg.mongo_uri, cfg.mongo_db)
    client.create_dictionary(cfg.exp_name, cfg.d_sae, cfg.exp_series)
    for i in range(len(result["index"])):
        client.update_feature(cfg.exp_name, result["index"][i].item(), {
            "act_times": result["act_times"][i].item(),
            "max_feature_acts": result["max_feature_acts"][i].item(),
            "feature_acts_all": result["feature_acts_all"][i].cpu().numpy(),
            "analysis": [
                {
                    "name": v["name"],
                    "feature_acts": v["feature_acts"][i].cpu().numpy(),
                    "contexts": v["contexts"][i].cpu().numpy(),
                } for v in result["analysis"]
            ]
        }, dictionary_series=cfg.exp_series)

    return result

@torch.no_grad()
def features_to_logits_runner(cfg: FeaturesDecoderConfig):
    print(cfg.exp_name + ' is running')
    sae = SparseAutoEncoder(cfg=cfg)
    if cfg.from_pretrained_path is not None:
        sae.load_state_dict(torch.load(cfg.from_pretrained_path, map_location=cfg.device)["sae"], strict=cfg.strict_loading)
    
    hf_model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir=cfg.cache_dir, local_files_only=cfg.local_files_only)
    model = HookedTransformer.from_pretrained('gpt2', device=cfg.device, cache_dir=cfg.cache_dir, hf_model=hf_model)
    model.eval()
    
    result_dict = features_to_logits(sae, model, cfg)
    
    client = MongoClient(cfg.mongo_uri, cfg.mongo_db)

    for feature_index, logits in result_dict.items():
        sorted_indeces = torch.argsort(logits)
        top_negative_logits = logits[sorted_indeces[:cfg.top]].cpu().tolist()
        top_positive_logits = logits[sorted_indeces[-cfg.top:]].cpu().tolist()
        top_negative_ids = sorted_indeces[:cfg.top].tolist()
        top_positive_ids = sorted_indeces[-cfg.top:].tolist()
        top_negative_tokens = model.to_str_tokens(torch.tensor(top_negative_ids), prepend_bos=False)
        top_positive_tokens = model.to_str_tokens(torch.tensor(top_positive_ids), prepend_bos=False)
        counts, edges = torch.histogram(logits.cpu(), bins=60, range=(-60.0, 60.0)) # Why logits.cpu():Could not run 'aten::histogram.bin_ct' with arguments from the 'CUDA' backend
        client.update_feature(cfg.exp_name, int(feature_index), {
            "logits": {
                "top_negative": [
                    {
                        "token_id": id,
                        "logit": logit,
                        "token": token
                    } for id, logit, token in zip(top_negative_ids, top_negative_logits, top_negative_tokens)
                ],
                "top_positive": [
                    {
                        "token_id": id,
                        "logit": logit,
                        "token": token
                    } for id, logit, token in zip(top_positive_ids, top_positive_logits, top_positive_tokens)
                ],
                "histogram": {
                    "counts": counts.cpu().tolist(),
                    "edges": edges.cpu().tolist()
                }
            }
        }, dictionary_series=cfg.exp_series)
    
    print(cfg.exp_name + ' done')

# def language_model_l0sae_runner(cfg: LanguageModelL0SAETrainingConfig):
#     if not cfg.use_ddp or cfg.rank == 0:
#         cfg.save_hyperparameters()
#         cfg.save_lm_config()
#     sae = L0SparseAutoEncoder(cfg=cfg)
#     if cfg.from_pretrained_path is not None:
#         sae.load_state_dict(torch.load(cfg.from_pretrained_path, map_location=cfg.device)["sae"], strict=cfg.strict_loading)
#     hf_model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir=cfg.cache_dir, local_files_only=cfg.local_files_only)
#     model = HookedTransformer.from_pretrained('gpt2', device=cfg.device, cache_dir=cfg.cache_dir, hf_model=hf_model)
#     model.eval()
#     activation_store = ActivationStore.from_config(model=model, cfg=cfg)

#     if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
#         wandb_run = wandb.init(project=cfg.wandb_project, config=cast(Any, cfg), name=cfg.run_name, entity=cfg.wandb_entity)
#         with open(os.path.join(cfg.exp_result_dir, cfg.exp_name, "train_wandb_id.txt"), "w") as f:
#             f.write(wandb_run.id)
#         wandb.watch(sae, log="all")

#     # train SAE
#     sae = l0train_sae(
#         model,
#         sae,
#         activation_store,
#         cfg,
#     )

#     if cfg.log_to_wandb and (not cfg.use_ddp or cfg.rank == 0):
#         wandb.finish()

#     return sae