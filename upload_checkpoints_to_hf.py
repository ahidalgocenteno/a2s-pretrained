from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi()

REPO_ID = "ahidalgocenteno/a2s-pretrained"

try:
    api.repo_info(repo_id=REPO_ID, repo_type="model")
    print(f"The repository {REPO_ID} already exists.")
except RepositoryNotFoundError:
    print(f"The repository {REPO_ID} does not exist. Creating it now as a public repository...")
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=False)

print("Uploading the checkpoint file...")
api.upload_file(
    path_or_fileobj="./weights/quartets_mert_330M/all-v1.ckpt",
    path_in_repo="checkpoints/quartets_mert_330M_transformer.ckpt",
    repo_id=REPO_ID,
    repo_type="model",
)

print("Uploading the configuration file...")
api.upload_file(
    path_or_fileobj="./weights/quartets_mert_330M/all_transformer_config.pkl",
    path_in_repo="checkpoints/quartets_mert_330M_transformer_config.pkl",
    repo_id=REPO_ID,
    repo_type="model",
)

print("All files have been uploaded successfully.")
