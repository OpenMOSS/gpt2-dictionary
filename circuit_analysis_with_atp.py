import json
import os
import sys
from typing import Any, Callable

sys.path.insert(0, os.getcwd())

from HookedTransformer import HookedTransformer
# from transformer_lens import HookedTransformer

from transformers import AutoModelForCausalLM

import networkx as nx
import random
import math
import pickle
import dataclasses
import numpy as np

from einops import repeat

import plotly.express as px
import torch
import torch.nn.functional as F
from core.config import SAEConfig
from core.sae import SparseAutoEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

hf_model = AutoModelForCausalLM.from_pretrained('gpt2')
model = HookedTransformer.from_pretrained('gpt2', device=device, hf_model=hf_model)
model.cfg.detach_pattern = True

def check_all_close():
	import transformer_lens
	origin_tl_model = transformer_lens.HookedTransformer.from_pretrained('gpt2', device=device, hf_model=hf_model)
	logits = model(model.to_tokens('Hello, World.'))
	origin_logits = origin_tl_model(origin_tl_model.to_tokens('Hello, World.'))
	assert torch.allclose(logits, origin_logits, atol=1e-4), f"Logits are not close: {logits} != {origin_logits}"

# input = model.to_tokens(" OpenMoss! OpenMoss! OpenMoss!", prepend_bos=False)
# input = model.to_tokens("Outside [Inside] Outside", prepend_bos=False)
# input = model.to_tokens("0 0 [1 1 1 [2] 3] 4", prepend_bos=False)
# input = model.to_tokens("Video in WebM support: Your browser doesn't support HTML5 video in WebM.", prepend_bos=False)
# input = model.to_tokens("Form-fitting TrekDry helps keep hands cool and comfortable. Form-fitting TrekDry material is lightweight and breathable.", prepend_bos=False)
# input = model.to_tokens(" it was its command line interface. You get so much leverage by being able to scaffold a [Inner Inner] A B A", prepend_bos=False)
# input = model.to_tokens("[[[ OpenMoss ]]] OpenMoss Open Moss ]", prepend_bos=False)
# input = model.to_tokens("Fruits:\n\napple red\n\nbanana yellow\n\ngrape purple", prepend_bos=False)
# input = model.to_tokens("Fruits:\n\nbanana yellow\n\napple red\n\ngrape purple", prepend_bos=False)
# input = model.to_tokens("You’re used to endlessly circular debates where Republican shills and Democratic shills", prepend_bos=False)
# input = model.to_tokens("Afterwards, Alice and Tom went to the shop. Tom gave a bunch of flowers to", prepend_bos=False)
# input = model.to_tokens("Afterwards, Tom and Alice went to the shop. Tom gave a bunch of flowers to", prepend_bos=False)
input = model.to_tokens("When Mary and John went to the store, Mary gave a bottle of milk to", prepend_bos=False)
# input = model.to_tokens("When Mary and John went to the store, John gave a bottle of milk to", prepend_bos=False)
# input = model.to_tokens("When John and Mary went to the store, John gave a bottle of milk to", prepend_bos=False)
# input = model.to_tokens("20 Parts Rosemary, 8 Parts Grapefruit", prepend_bos=False)

answer = model.to_tokens(" John", prepend_bos=False)
assert answer.size(0) == 1
logit = model(input)[0, -1, answer.item()]

logit.backward()

for block in model.blocks:
	print(block.mlp_sae)
	print(block.attn_sae)


