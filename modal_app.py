"""Modal app for olens-1layer: oracle lens whose verbalizer (AV) is a ONE-LAYER transformer.
Separate workspace from olens-new-arch: own app + volume; reads the shared harvest + 27B weights
READ-ONLY from the olens-new-arch volume (no writes there).

    modal run --detach modal_app.py --task train --gpus 2 --nproc 2 --script onelayer_av.py --args "..."
"""
import os
import subprocess

import modal

APP_NAME = os.environ.get("OL1_APP", "olens-1layer")
GPU_TYPE = os.environ.get("OL1_GPU", "B200")
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_LOCAL = os.path.join(HERE, "src")
SRC_REMOTE = "/root/src"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.8.0", "transformers==5.5.4", "peft==0.19.1", "accelerate", "safetensors",
                 "sentencepiece", "pyarrow", "numpy", "pandas", "wandb", "einops", "scipy", "pyyaml",
                 "huggingface_hub[hf_transfer]", "flash-linear-attention")
    .env({"HF_HOME": "/vol_data/hf_cache", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONUNBUFFERED": "1",
          "PYTHONPATH": SRC_REMOTE})
    .add_local_dir(SRC_LOCAL, SRC_REMOTE, copy=False, ignore=["__pycache__", "*.pyc"])
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name("olens-1layer", create_if_missing=True)
vol_data = modal.Volume.from_name("olens-new-arch")          # harvest + 27B weights, read-only
vol_maemm = modal.Volume.from_name("maemm-data")            # FineFineWeb token corpus, read-only
vol_modlens = modal.Volume.from_name("celeste-modlens-vol")   # modulation-lens session: AR adapter + head, amu (read-only)
VOLS = {"/vol": vol, "/vol_data": vol_data.read_only(), "/vol_maemm": vol_maemm.read_only(), "/vol_modlens": vol_modlens.read_only()}
SECRETS = [modal.Secret.from_name("maemm-hf"), modal.Secret.from_name("maemm-wandb")]


def _run(cmd, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    print("[run]", cmd, flush=True)
    p = subprocess.run(cmd, shell=True, cwd=SRC_REMOTE if os.path.isdir(SRC_REMOTE) else "/root", env=env)
    vol.commit()
    print(f"[run] exit {p.returncode}", flush=True)
    return p.returncode


@app.function(gpu=GPU_TYPE, volumes=VOLS, timeout=23 * 60 * 60, secrets=SECRETS)
def train(script: str, args: str = "", nproc: int = 1):
    cmd = (f"torchrun --nproc_per_node {nproc} {SRC_REMOTE}/{script} {args}" if nproc > 1
           else f"python {SRC_REMOTE}/{script} {args}")
    return _run(cmd)


@app.function(volumes=VOLS, timeout=2 * 60 * 60, cpu=4.0, memory=32768, secrets=SECRETS)
def pyrun(code: str):
    import textwrap
    open("/tmp/snippet.py", "w").write(textwrap.dedent(code))
    return _run("python /tmp/snippet.py")


@app.local_entrypoint()
def main(task: str, script: str = "", args: str = "", code: str = "", gpus: int = 1, nproc: int = 0, mem_gb: int = 0):
    if task == "train":
        opts = {"gpu": f"{GPU_TYPE}:{gpus}"}
        if mem_gb:
            opts["memory"] = mem_gb * 1024
        h = train.with_options(**opts).spawn(script, args, nproc or gpus)
        print(f"SPAWNED {h.object_id}", flush=True)
    elif task == "train-many":                     # several spawns in ONE app (workspace cap: 100 ephemeral apps)
        opts = {"gpu": f"{GPU_TYPE}:{gpus}"}
        if mem_gb:
            opts["memory"] = mem_gb * 1024
        fn = train.with_options(**opts)
        for a in [x.strip() for x in args.split(";;") if x.strip()]:
            h = fn.spawn(script, a, nproc or gpus)
            print(f"SPAWNED {h.object_id} :: {a[:60]}", flush=True)
    elif task == "pyrun":
        h = pyrun.spawn(code)
        print(f"SPAWNED {h.object_id}", flush=True)
    else:
        raise SystemExit(f"unknown task {task}")
