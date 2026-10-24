"""Upload the curated train/test folders to a private HF dataset repo.

Token comes from HF_TOKEN in the environment or in .env; it is never printed.
Usage: python src/push_data.py --tag 2026-10-24 [--user NAME]
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def read_token() -> str:
    token = os.environ.get("HF_TOKEN")
    env_file = ROOT / ".env"
    if not token and env_file.exists():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN":
                token = value.strip().strip("'\"")
    if not token:
        raise SystemExit("HF_TOKEN not found in the environment or .env")
    return token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--user", default="bharataggarwal56")
    ap.add_argument("--name", default="ticketforge-data")
    args = ap.parse_args()

    api = HfApi(token=read_token())
    who = api.whoami()["name"]
    if who != args.user:
        raise SystemExit(f"token belongs to {who!r}, expected {args.user!r}; not uploading")

    repo_id = f"{args.user}/{args.name}"
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    for kind in ("train", "test"):
        folder = f"{kind}_{args.tag}"
        local = ROOT / "data" / "curated" / folder
        if not local.is_dir():
            raise SystemExit(f"missing {local}")
        api.upload_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=str(local),
            path_in_repo=folder, commit_message=f"Add {folder}",
        )

    info = api.dataset_info(repo_id)
    files = sorted(api.list_repo_files(repo_id, repo_type="dataset"))
    print(f"{repo_id} private={info.private}")
    print("\n".join(files))


if __name__ == "__main__":
    main()
