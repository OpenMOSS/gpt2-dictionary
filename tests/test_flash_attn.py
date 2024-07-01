from typing import Any, cast
import os
import sys
sys.path.insert(0, os.getcwd())

import wandb
import logging
import random

import torch

from flash_attn import flash_attn_func
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_gpt2_weights

from lm_saes.config import (
    ActivationGenerationConfig,
    LanguageModelConfig,
    LanguageModelSAEAnalysisConfig,
    LanguageModelSAETrainingConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAEPruningConfig,
    FeaturesDecoderConfig,
)
from lm_saes.database import MongoClient
from lm_saes.evals import run_evals
from lm_saes.sae import SparseAutoEncoder
from lm_saes.activation.activation_dataset import make_activation_dataset
from lm_saes.activation.activation_store import ActivationStore
from lm_saes.sae_training import prune_sae, train_sae
from lm_saes.analysis.sample_feature_activations import sample_feature_activations
from lm_saes.analysis.features_to_logits import features_to_logits
from lm_saes.activation.activation_source import TokenActivationSource
from lm_saes.activation.token_source import TokenSource

from datasets import load_dataset, load_from_disk
from transformer_lens import HookedTransformer

import pytest
HOOK_SUFFIX={"mlp":"hook_mlp_out", "self_attn":"hook_attn_out", "resid":"hook_resid_post"}

@pytest.fixture
def dataset():
    return load_dataset("Skylion007/openwebtext", split="train")

@pytest.fixture
def dataloader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=32)

@pytest.fixture
def model():
    return HookedTransformer.from_pretrained('gpt2')

def pytest_generate_tests(metafunc):
    dataset = load_from_disk("/remote-home/share/research/mechinterp/gpt2-dictionary/data/openwebtext")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    if 'test_input' in metafunc.fixturenames:
        test_input = []
        for _ in range(10):
            text = ''.join(next(iter(dataloader))['text'])
            idx = random.randrange(0, len(text)-32)
            test_input.append(text[idx:idx+32])
        metafunc.parametrize('test_input', test_input)

@pytest.fixture
def prepare_config(args):
    cfg = LanguageModelConfig.from_flattened(dict(
        # LanguageModelConfig
        model_name = args['model_name'],                            # The model name or path for the pre-trained model.
        model_from_pretrained_path = args['model_path'],
        d_model = args['d_model'],                                  # The hidden size of the model.
        
        # RunnerConfig
        device = "cuda",                                # The device to place all torch tensors.
        seed = 42,                                      # The random seed.
        dtype = torch.bfloat16,                          # The torch data type of non-integer tensors.

        exp_name = f"test",
        exp_series = "default",
        exp_result_dir = "results",
    ))
    return cfg

@pytest.fixture
def prepare_llama3_models():
    model_path = "/remote-home/share/models/llama3_hf/Meta-Llama-3-8B"
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    hf_no_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                       attn_implementation="eager", 
                                                       cache_dir=None,
                                                       torch_dtype=torch.bfloat16, 
                                                       local_files_only=False)
    hf_no_model.eval()
    hf_fa_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                       attn_implementation="flash_attention_2", 
                                                       cache_dir=None,
                                                       torch_dtype=torch.bfloat16, 
                                                       local_files_only=False)
    hf_fa_model.eval()
    hf_fa_model.to(device)
    return hf_no_model, hf_fa_model

@pytest.fixture
def prepare_models(prepare_config):
    cfg = prepare_config
    hf_model = AutoModelForCausalLM.from_pretrained(
        (
            cfg.model_name
            if cfg.model_from_pretrained_path is None
            else cfg.model_from_pretrained_path
        ),
        cache_dir=cfg.cache_dir,
        local_files_only=cfg.local_files_only,
        torch_dtype=cfg.dtype,
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        (
            cfg.model_name
            if cfg.model_from_pretrained_path is None
            else cfg.model_from_pretrained_path
        ),
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=True,
    )

    # FlashAttention only allow dtype of bfp16 and fp16
    assert cfg.dtype in [torch.bfloat16, torch.float16] 

    fa_model = HookedTransformer.from_pretrained(
        cfg.model_name,
        use_flash_attn=True,
        device=cfg.device,
        cache_dir=cfg.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=cfg.dtype,
    )
    fa_model.eval()
    no_model = HookedTransformer.from_pretrained(
        cfg.model_name,
        use_flash_attn=False,
        device=cfg.device,
        cache_dir=cfg.cache_dir,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        dtype=cfg.dtype,
    )
    no_model.eval()
    logging.warning("Model loaded!")
    return fa_model, no_model, cfg


def test_language_model_sae_runner(prepare_models, prepare_llama3_models):
    # FIXME dataset path need to be removed
    dataset = load_from_disk("/remote-home/share/research/mechinterp/gpt2-dictionary/data/openwebtext")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    test_input_list = []
    for _ in range(10):
        text = ''.join(next(iter(dataloader))['text'])
        idx = random.randrange(0, len(text)-64)
        test_input_list.append(text[idx:idx+64])
    fa_model, no_model, cfg = prepare_models
    hf_no_model, hf_fa_model = prepare_llama3_models

    # bfloat16 dtype test
    if cfg.dtype == torch.bfloat16:
        for model in [fa_model, no_model]:
            for name, obj in vars(HookedTransformer).items():
                if isinstance(obj, property):
                    try:
                        param = model.__getattribute__(name)
                        assert (param.dtype == torch.bfloat16)
                    except:
                        logging.warning(f"Does not have attribute {name}")

    # current_tokens_n = 0
    # total_tokens_n = 1000_000
    layer_name = [f"model.layers.{i}" + e for e in [".self_attn", '.mlp', ''] for i in range(no_model.cfg.n_layers)]+['lm_head']
    hf_no_cache = {'self_attn':[], 'mlp':[], 'resid':[]}
    hf_no_handle = []
    hf_fa_cache = {'self_attn':[], 'mlp':[], 'resid':[]}
    hf_fa_handle = []
    def no_attn_hook_fn(module, input, output):
        hf_no_cache['self_attn'].append(output[0])
    def no_mlp_hook_fn(module, input, output):
        if isinstance(output, tuple):
            hf_no_cache['mlp'].append(output[0])
        else:
            hf_no_cache['mlp'].append(output)
    def no_resid_hook_fn(module, input, output):
        if isinstance(output, tuple):
            hf_no_cache['resid'].append(output[0])
        else:
            hf_no_cache['resid'].append(output)
    def fa_attn_hook_fn(module, input, output):
        hf_fa_cache['self_attn'].append(output[0].cpu())
    def fa_mlp_hook_fn(module, input, output):
        if isinstance(output, tuple):
            hf_fa_cache['mlp'].append(output[0].cpu())
        else:
            hf_fa_cache['mlp'].append(output.cpu())
    def fa_resid_hook_fn(module, input, output):
        if isinstance(output, tuple):
            hf_fa_cache['resid'].append(output[0].cpu())
        else:
            hf_fa_cache['resid'].append(output.cpu())
    for (name, module) in hf_no_model.named_modules():
        if name in layer_name:
            if "self_attn" in name:
                hf_no_handle.append(module.register_forward_hook(hook=no_attn_hook_fn))
            elif "mlp" in name:
                hf_no_handle.append(module.register_forward_hook(hook=no_mlp_hook_fn))
            else:
                hf_no_handle.append(module.register_forward_hook(hook=no_resid_hook_fn))
    for (name, module) in hf_fa_model.named_modules():
        if name in layer_name:
            if "self_attn" in name:
                hf_fa_handle.append(module.register_forward_hook(hook=fa_attn_hook_fn))
            elif "mlp" in name:
                hf_fa_handle.append(module.register_forward_hook(hook=fa_mlp_hook_fn))
            else:
                hf_fa_handle.append(module.register_forward_hook(hook=fa_resid_hook_fn))
    
    # batch = next(iter((dataloader)))
    for test_input in test_input_list:
        tokens = no_model.to_tokens(test_input, prepend_bos=not False).to(cfg.device)
        print("Preparation done!")
        # import pdb
        # pdb.set_
        fa_logits, fa_cache = fa_model.run_with_cache(tokens, return_type="logits")
        no_logits, no_cache = no_model.run_with_cache(tokens, return_type="logits")
        _ = hf_no_model(tokens.cpu(), use_cache=False)
        _ = hf_fa_model(tokens, use_cache=False)
        for layer in range(no_model.cfg.n_layers):
            # q = no_cache['blocks.0.attn.hook_rot_q']
            # k = no_cache['blocks.0.attn.hook_rot_k']
            # v = no_cache['blocks.0.attn.hook_v']
            # k_repeated = k.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
            # v_repeated = v.repeat_interleave(q.shape[2] // k.shape[2], dim=2)
            # fa_z_t = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2), k_repeated.transpose(1,2), v_repeated.transpose(1,2), is_causal=True).transpose(1,2)
            # fa_z_f = flash_attn_func(q, k_repeated, v_repeated, causal=True)
            # fa_z_f_gqa = flash_attn_func(q, k, v, causal=True)
            # z = no_cache['blocks.0.attn.hook_z']
            # import pdb 
            # pdb.set_trace()
            for abbr, hook_suffix_abbr in HOOK_SUFFIX.items():
                fa_value = fa_cache[f'blocks.{layer}.{hook_suffix_abbr}']
                no_value = no_cache[f'blocks.{layer}.{hook_suffix_abbr}']

                hf_fa_value = hf_fa_cache[abbr][layer]
                hf_no_value = hf_no_cache[abbr][layer]

                delta_max_fa_no = torch.abs(fa_value - no_value).max().item()
                delta_max_hf_fa_no = torch.abs(hf_fa_value - hf_no_value).max().item()
                logging.warning(f"L{layer}{abbr}\ttl:{delta_max_fa_no}\thf:{delta_max_hf_fa_no}")
                assert (delta_max_fa_no < delta_max_hf_fa_no * 5)
        d_logits_fa_no = torch.abs(fa_logits - no_logits).max().item()
        d_logits_hf_fa_no = torch.abs(hf_fa_cache['resid'][-1] - hf_no_cache['resid'][-1]).max().item()

        logging.warning(f"Logits\ttl:{d_logits_fa_no}\thf:{d_logits_hf_fa_no}")
        assert (d_logits_fa_no < d_logits_hf_fa_no * 5)
        for e1, e2 in zip(hf_no_handle, hf_fa_handle):
            e1.remove()
            e2.remove()