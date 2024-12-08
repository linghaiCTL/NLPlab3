import huggingface_hub
import os

# set mirror

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
repo_id = "google-research-datasets/mbpp"
repo_type = "dataset"
save_dir = "datasets/mbpp"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
huggingface_hub.snapshot_download(repo_id, repo_type=repo_type, local_dir=save_dir)