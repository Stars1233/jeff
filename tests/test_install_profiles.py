"""The serving-only install (`uv sync --no-default-groups`) must be enough to run jeff-serve
(https://github.com/firelex/jeff/issues/2)."""

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ONLY = {"datasets", "matplotlib", "accelerate", "dotenv", "imageio", "imageio_ffmpeg", "fla", "kernels", "vizdoom"}


def test_the_server_imports_no_training_packages() -> None:
    code = ("import sys, jeff.server; "
            f"print(sorted({{m.split('.')[0] for m in sys.modules}} & set({sorted(TRAINING_ONLY)!r})))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT)
    assert result.stdout.strip() == "[]", f"jeff.server imports training-only packages: {result.stdout.strip()}"


def test_the_core_install_is_serving_only_and_uv_sync_still_installs_everything() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    core = {requirement.split("==")[0].split(">=")[0] for requirement in project["project"]["dependencies"]}
    assert core == {"torch", "torchvision", "transformers", "pillow", "fastapi", "uvicorn", "safetensors", "numpy",
                    "huggingface-hub"}
    train = {requirement.split("==")[0] for requirement in project["dependency-groups"]["train"]}
    assert {"datasets", "accelerate", "httpx"} <= train
    assert project["tool"]["uv"]["default-groups"] == ["dev", "train"]
