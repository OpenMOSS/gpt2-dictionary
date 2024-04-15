import os

import numpy as np
import torch

from transformers import GPT2Tokenizer, AutoModelForCausalLM

from transformer_lens import HookedTransformer

from datasets import Dataset

import msgpack

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

import plotly.express as px
import plotly.graph_objects as go

from core.analysis.auto_interp import check_description, generate_description
from core.config import AutoInterpConfig, LanguageModelConfig, SAEConfig
from core.database import MongoClient
from core.sae import SparseAutoEncoder

result_dir = os.environ.get("RESULT_DIR", "results")
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

app.add_middleware(GZipMiddleware, minimum_size=1000)

client = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017"), os.environ.get("MONGO_DB", "mechinterp"))
dictionary_series = os.environ.get("DICTIONARY_SERIES", None)
ckpt_name = os.environ.get("DICTIONARY_CKPT_NAME", "final.pt")

tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
model = HookedTransformer.from_pretrained("gpt2", device=device, hf_model=hf_model)
model.eval()

sae_cache = {}
lm_cache = {}

def get_sae(dictionary_name: str) -> SparseAutoEncoder:
    if dictionary_name not in sae_cache:
        cfg = SAEConfig(
            **SAEConfig.get_hyperparameters(
                dictionary_name, result_dir, ckpt_name, True
            ),
            # RunnerConfig
            use_ddp=False,
            device=device,
            seed=42,
            dtype=torch.float32,
            exp_name=dictionary_name,
        )
        sae = SparseAutoEncoder(cfg=cfg)
        sae.load_state_dict(
            torch.load(cfg.from_pretrained_path, map_location=cfg.device)["sae"]
        )
        sae.eval()
        sae_cache[dictionary_name] = sae
    return sae_cache[dictionary_name]


def make_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


@app.get("/dictionaries")
def list_dictionaries():
    return client.list_dictionaries(dictionary_series=dictionary_series)


@app.get("/dictionaries/{dictionary_name}/features/{feature_index}")
def get_feature(dictionary_name: str, feature_index: str):
    if isinstance(feature_index, str):
        if feature_index == "random":
            feature = client.get_random_alive_feature(dictionary_name, dictionary_series=dictionary_series)
        else:
            try:
                feature_index = int(feature_index)
            except ValueError:
                return Response(
                    content=f"Feature index {feature_index} is not a valid integer",
                    status_code=400,
                )
            feature = client.get_feature(dictionary_name, feature_index, dictionary_series=dictionary_series)

    if feature is None:
        return Response(
            content=f"Feature {feature_index} not found in dictionary {dictionary_name}",
            status_code=404,
        )

    sample_groups = []
    for analysis in feature["analysis"]:
        samples = [
            {
                "context": [
                    bytearray([tokenizer.byte_decoder[c] for c in t])
                    for t in tokenizer.convert_ids_to_tokens(analysis["contexts"][i])
                ],
                "feature_acts": analysis["feature_acts"][i],
            }
            for i in range(len(analysis["feature_acts"]))
        ]
        sample_groups.append(
            {
                "analysis_name": analysis["name"],
                "samples": samples,
            }
        )

    feature_activation_histogram = px.histogram(feature["feature_acts_all"], width=600, nbins=50)

    feature_activation_histogram = go.Histogram(
        x=feature["feature_acts_all"],
        nbinsx=50,
        hovertemplate="Count: %{y}<br>Range: %{x}<extra></extra>",
        marker_color="#636EFA",
        showlegend=False,
    ).to_plotly_json()

    if "logits" in feature:
        logits_bin_edges = np.array(feature["logits"]["histogram"]["edges"])
        logits_histogram = go.Bar(
            x=(logits_bin_edges[:-1] + logits_bin_edges[1:]) / 2,
            customdata=np.dstack([logits_bin_edges[:-1], logits_bin_edges[1:]]).squeeze(),
            y=np.array(feature["logits"]["histogram"]["counts"]),
            hovertemplate="Count: %{y}<br>Range: %{customdata[0]} - %{customdata[1]}<extra></extra>",
            marker_color=["#EF553B" for _ in range((len(logits_bin_edges) - 1) // 2)] + ["#636EFA" for _ in range((len(logits_bin_edges) - 1) // 2)],
            showlegend=False,
        ).to_plotly_json()

    return Response(
        content=msgpack.packb(
            make_serializable(
                {
                    "feature_index": feature["index"],
                    "dictionary_name": dictionary_name,
                    "feature_activation_histogram": [feature_activation_histogram],
                    "act_times": feature["act_times"],
                    "max_feature_act": feature["max_feature_acts"],
                    "sample_groups": sample_groups,
                    "logits": {
                        "top_positive": list(reversed(feature["logits"]["top_positive"])),
                        "top_negative": feature["logits"]["top_negative"],
                        "histogram": [logits_histogram],
                    } if "logits" in feature else None,
                    "interpretation": feature["interpretation"] if "interpretation" in feature else None,
                }
            )
        ),
        media_type="application/x-msgpack",
    )

@app.get("/dictionaries/{dictionary_name}")
def get_dictionary(dictionary_name: str):
    feature_activation_times = client.get_feature_act_times(dictionary_name, dictionary_series=dictionary_series)
    if feature_activation_times is None:
        return Response(
            content=f"Dictionary {dictionary_name} not found", status_code=404
        )
    log_act_times = np.log10(np.array(list(feature_activation_times.values())))
    feature_activation_times_histogram = go.Histogram(
        x=log_act_times,
        nbinsx=100,
        hovertemplate="Count: %{y}<br>Range: %{x}<extra></extra>",
        marker_color="#636EFA",
        showlegend=False,
    ).to_plotly_json()

    alive_feature_count = client.get_alive_feature_count(dictionary_name, dictionary_series=dictionary_series)
    if alive_feature_count is None:
        return Response(
            content=f"Dictionary {dictionary_name} not found", status_code=404
        )

    return Response(
        content=msgpack.packb(
            make_serializable(
                {
                    "dictionary_name": dictionary_name,
                    "feature_activation_times_histogram": [feature_activation_times_histogram],
                    "alive_feature_count": alive_feature_count,
                }
            )
        ),
        media_type="application/x-msgpack",
    )



@app.post("/dictionaries/{dictionary_name}/features/{feature_index}/custom")
def feature_activation_custom_input(
    dictionary_name: str, feature_index: int, input_text: str
):
    try:
        sae = get_sae(dictionary_name)
        hook_point = LanguageModelConfig.get_lm_config(dictionary_name, result_dir)["hook_point"]
    except FileNotFoundError:
        return Response(
            content=f"Dictionary {dictionary_name} not found", status_code=404
        )

    if feature_index < 0 or feature_index >= sae.cfg.d_sae:
        return Response(
            content=f"Feature index {feature_index} is out of range", status_code=400
        )

    with torch.no_grad():
        input = model.to_tokens(input_text, prepend_bos=False)
        _, cache = model.run_with_cache(input, names_filter=[hook_point])
        activation = cache[hook_point][0]

        _, (_, aux) = sae(activation)
        feature_acts = aux["feature_acts"]
        sample = {
            "context": [
                bytearray([tokenizer.byte_decoder[c] for c in t])
                for t in tokenizer.convert_ids_to_tokens(input[0])
            ],
            "feature_acts": feature_acts[:, feature_index].cpu().numpy().tolist(),
        }

    return Response(content=msgpack.packb(sample), media_type="application/x-msgpack")

@app.post("/dictionaries/{dictionary_name}/custom")
def dictionary_custom_input(dictionary_name: str, input_text: str):
    try:
        sae = get_sae(dictionary_name)
        hook_point = LanguageModelConfig.get_lm_config(dictionary_name, result_dir)["hook_point"]
    except FileNotFoundError:
        return Response(
            content=f"Dictionary {dictionary_name} not found", status_code=404
        )
    
    max_feature_acts = client.get_max_feature_acts(dictionary_name, dictionary_series=dictionary_series)

    with torch.no_grad():
        input = model.to_tokens(input_text, prepend_bos=False)
        _, cache = model.run_with_cache(input, names_filter=[hook_point])
        activation = cache[hook_point][0]

        _, (_, aux) = sae(activation)
        feature_acts = aux["feature_acts"]
        sample = {
            "context": [
                bytearray([tokenizer.byte_decoder[c] for c in t])
                for t in tokenizer.convert_ids_to_tokens(input[0])
            ],
            "feature_acts_indices": [
                feature_acts[i].nonzero(as_tuple=True)[0].cpu().numpy().tolist()
                for i in range(feature_acts.shape[0])
            ],
            "feature_acts": [
                feature_acts[i][feature_acts[i].nonzero(as_tuple=True)[0]].cpu().numpy().tolist()
                for i in range(feature_acts.shape[0])
            ],
            "max_feature_acts": [
                [max_feature_acts[j] for j in feature_acts[i].nonzero(as_tuple=True)[0].cpu().numpy().tolist()]
                for i in range(feature_acts.shape[0])
            ]
        }

    return Response(content=msgpack.packb(sample), media_type="application/x-msgpack")


@app.post("/dictionaries/{dictionary_name}/features/{feature_index}/interpret")
def feature_interpretation(
    dictionary_name: str,
    feature_index: int,
    type: str,
    custom_interpretation: str | None = None,
):
    if type == "custom":
        interpretation = {
            "text": custom_interpretation,
            "validation": [
                {
                    "method": "manual",
                    "passed": True,
                }
            ],
        }
    elif type == "auto":
        cfg = AutoInterpConfig(
            **{
                **SAEConfig.get_hyperparameters(
                    dictionary_name, result_dir, ckpt_name, True
                ),
                **LanguageModelConfig.get_lm_config(dictionary_name, result_dir),
                "openai_api_key": os.environ.get("OPENAI_API_KEY"),
                "openai_base_url": os.environ.get("OPENAI_BASE_URL"),
            }
        )
        feature = client.get_feature(dictionary_name, feature_index, dictionary_series=dictionary_series)
        result = generate_description(model, feature["analysis"][0], cfg)
        interpretation = {
            "text": result["response"],
            "validation": [],
            "detail": result,
        }
    elif type == "validate":
        cfg = AutoInterpConfig(
            **{
                **SAEConfig.get_hyperparameters(
                    dictionary_name, result_dir, ckpt_name, True
                ),
                **LanguageModelConfig.get_lm_config(dictionary_name, result_dir),
                "openai_api_key": os.environ.get("OPENAI_API_KEY"),
                "openai_base_url": os.environ.get("OPENAI_BASE_URL"),
            }
        )
        feature = client.get_feature(dictionary_name, feature_index, dictionary_series=dictionary_series)
        interpretation = feature["interpretation"] if "interpretation" in feature else None
        if interpretation is None:
            return Response(content="Feature interpretation not found", status_code=404)
        validation = interpretation["validation"]
        if not any(v["method"] == "activation" for v in validation):
            validation_result = check_description(
                model,
                cfg,
                feature_index,
                interpretation["text"],
                False,
                feature_activation=feature["analysis"][0],
            )
            validation.append(
                {
                    "method": "activation",
                    "passed": validation_result["passed"],
                    "detail": validation_result,
                }
            )
        if not any(v["method"] == "generative" for v in validation):
            validation_result = check_description(
                model,
                cfg,
                feature_index,
                interpretation["text"],
                True,
                sae=get_sae(dictionary_name),
            )
            validation.append(
                {
                    "method": "generative",
                    "passed": validation_result["passed"],
                    "detail": validation_result,
                }
            )

    try:
        client.update_feature(dictionary_name, feature_index, 
            {"interpretation": interpretation}, dictionary_series=dictionary_series)
    except ValueError as e:
        return Response(content=str(e), status_code=400)
    return interpretation


@app.get("/attn_heads/{layer}/{head}")
def get_attn_head(layer: int, head: int):
    attn_head = client.get_attn_head(layer, head, dictionary_series=dictionary_series)
    if attn_head is None:
        return Response(
            content=f"Attention head {layer}/{head} not found", status_code=404
        )
    attn_scores = [{
        "dictionary1_name": v["dictionary1"]["name"],
        "dictionary2_name": v["dictionary2"]["name"],
        "top_attn_scores": v["top_attn_scores"],
    } for v in attn_head["attn_scores"]]

    return {
        "layer": layer,
        "head": head,
        "attn_scores": attn_scores,
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
