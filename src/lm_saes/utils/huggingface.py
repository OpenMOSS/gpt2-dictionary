# Some helper function to interact with huggingface API

from typing import Optional, Union
import os
import shutil
from huggingface_hub import create_repo, upload_folder, snapshot_download


def upload_pretrained_sae_to_hf(sae_path: str, repo_id: str, private: bool = False):
    """Upload a pretrained SAE model to huggingface model hub

    Args:
        sae_path (str): path to the local SAE model
    """

    from lm_saes.config import LanguageModelConfig
    from lm_saes.sae import SparseAutoEncoder

    # Load the model
    sae = SparseAutoEncoder.from_pretrained(sae_path)
    lm_config = LanguageModelConfig.from_pretrained_sae(sae_path)

    # Create local temporary directory for uploading
    folder_name = sae.cfg.hook_point_in if sae.cfg.hook_point_in == sae.cfg.hook_point_out else f"{sae.cfg.hook_point_in}-{sae.cfg.hook_point_out}"
    os.makedirs(f"{sae_path}/{folder_name}", exist_ok=True)

    try:
        # Save the model
        create_repo(repo_id=repo_id, private=private, exist_ok=True)

        sae.save_pretrained(f"{sae_path}/{folder_name}")
        sae.cfg.save_hyperparameters(f"{sae_path}/{folder_name}")
        lm_config.save_lm_config(f"{sae_path}/{folder_name}")

        # Upload the model
        upload_folder(folder_path=f"{sae_path}/{folder_name}", repo_id=repo_id, path_in_repo=folder_name, commit_message=f"Upload pretrained SAE model. Hook point: {folder_name}. Language Model Name: {lm_config.model_name}")

    finally:
        # Remove the temporary directory
        shutil.rmtree(f"{sae_path}/{folder_name}")

def download_pretrained_sae_from_hf(repo_id: str, hook_point: str):
    """Download a pretrained SAE model from huggingface model hub

    Args:
        repo_id (str): id of the repo
        hook_point (str): hook point
    """

    snapshot_path = snapshot_download(repo_id=repo_id, allow_patterns=[f"{hook_point}/*"])
    return os.path.join(snapshot_path, hook_point)

def parse_pretrained_name_or_path(pretrained_name_or_path: str):
    if os.path.exists(pretrained_name_or_path):
        return pretrained_name_or_path
    else:
        repo_id = "/".join(pretrained_name_or_path.split("/")[:2])
        hook_point = "/".join(pretrained_name_or_path.split("/")[2:])
        return download_pretrained_sae_from_hf(repo_id, hook_point)