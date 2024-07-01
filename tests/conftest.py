import torch
import pytest

from lm_saes.config import LanguageModelConfig
from lm_saes.runner import language_model_sae_runner

def pytest_addoption(parser):
    parser.addoption("--layer", nargs="*", type=int, required=False, help='Layer number')
    parser.addoption("--batch_size", type=int, required=False, default=4096, help='Batchsize, default 4096')
    parser.addoption("--lr", type=float, required=False, default=8e-5, help='Learning rate, default 8e-5')
    parser.addoption("--expdir", type=str, required=False, default="path/to/results", help='Export directory, default path')
    parser.addoption("--useddp", type=bool, required=False, default=False, help='If using distributed method, default False')
    parser.addoption('--attn_type', type=str, required=False, choices=['flash', 'normal'], default="flash", help='Use or not use log of wandb, default True')
    parser.addoption('--dtype', type=str, required=False, choices=['fp32', 'bfp16'], default="fp32", help='Dtype, default fp32')
    parser.addoption('--model_name', type=str, required=False, default="meta-llama/Meta-Llama-3-8B", help='Supported model name of TransformerLens, default gpt2')
    parser.addoption('--d_model', type=int, required=False, default=4096, help='Dimension of model hidden states, default 4096')
    parser.addoption('--model_path', type=str, required=False, default="path/to/model", help='Hugging-face model path used to load.')
 
@pytest.fixture
def args(request):
    return {"layer":request.config.getoption("--layer"),
            "batch_size":request.config.getoption("--batch_size"),
            "lr":request.config.getoption("--lr"),
            "expdir":request.config.getoption("--expdir"),
            "useddp":request.config.getoption("--useddp"),
            "attn_type":request.config.getoption("--attn_type"),
            "dtype":request.config.getoption("--dtype"),
            "model_name":request.config.getoption("--model_name"),
            "model_path":request.config.getoption("--model_path"),
            "d_model":request.config.getoption("--d_model"),
            }
