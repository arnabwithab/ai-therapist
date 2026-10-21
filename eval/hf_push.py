"""One-file uploads to a Hugging Face dataset repo (eval outputs)."""

import logging
import os

from huggingface_hub import HfApi

log = logging.getLogger(__name__)


def push_file(local_path, repo, remote_path):
    """Upload local_path to repo:remote_path. No-op when repo is empty.

    Creates the (private) dataset repo on first use; fails fast without HF_TOKEN.
    """
    if not repo:
        return
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF push requested (--hf-repo) but HF_TOKEN is not set")
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=repo,
        repo_type="dataset",
    )
    log.info("pushed %s -> %s:%s", local_path, repo, remote_path)
