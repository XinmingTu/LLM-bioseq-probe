"""scripts/probe/protein_lm.py on Modal GPUs, one model per GPU -> results/plm_confirm/. About $1.

The weights are public on Hugging Face (biohub/ESMC-600M, biohub/esm3-sm-open-v1, AI4PD/ProtGPT3-*),
so no token is needed.

    modal run scripts/probe/protein_lm_modal.py --runs esmc-600m:prefix,esm3-open:prefix,protgpt3-1.3B:prefix,protgpt3-10B:prefix
    modal run scripts/probe/protein_lm_modal.py --runs protgpt3-10B:prefix --sequences data/sequences_pilot.tsv --out results/plm_pilot
    modal run scripts/probe/protein_lm_modal.py --runs progen3-3b:prefix   # ProGen3 runs in its own image
"""

from pathlib import Path

import modal

app = modal.App("llm-bioseq-probe-plm")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("esm==3.4.3", "transformers", "pandas")
    .add_local_file("scripts/probe/protein_lm.py", "/work/protein_lm.py")
    .add_local_file("data/confirm_sequences.tsv", "/work/data/confirm_sequences.tsv")
    .add_local_file("data/sequences_pilot.tsv", "/work/data/sequences_pilot.tsv")
    .add_local_file("data/small_family_sequences.tsv", "/work/data/small_family_sequences.tsv")
    .add_local_file("data/d_set_sequences.tsv", "/work/data/d_set_sequences.tsv")
)


@app.function(image=image, gpu=["L40S", "A100", "A100-80GB", "H100"], timeout=3 * 3600)
def run(spec: str, sequences: str, out: str) -> dict[str, str]:
    import os
    import sys

    os.chdir("/work")
    sys.path.insert(0, "/work")
    import protein_lm

    protein_lm.main(spec, sequences, out)
    return {p.name: p.read_text() for p in Path(out).glob("*.jsonl")}


progen3_image = (  # Profluent's image: torch 2.5.1, flash-attn and megablocks built for its Python 3.11
    modal.Image.from_registry("ghcr.io/profluent-ai/progen3:v0.1.0",
                              setup_dockerfile_commands=["RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python"])
    .pip_install("git+https://github.com/Profluent-AI/progen3.git")
    .add_local_file("scripts/probe/protein_lm.py", "/work/protein_lm.py")
    .add_local_file("data/confirm_sequences.tsv", "/work/data/confirm_sequences.tsv")
    .add_local_file("data/sequences_pilot.tsv", "/work/data/sequences_pilot.tsv")
    .add_local_file("data/small_family_sequences.tsv", "/work/data/small_family_sequences.tsv")
    .add_local_file("data/d_set_sequences.tsv", "/work/data/d_set_sequences.tsv")
)


@app.function(image=progen3_image, gpu=["A100-80GB", "H100", "A100"], timeout=3 * 3600)
def run_progen3(spec: str, sequences: str, out: str) -> dict[str, str]:
    return run.local(spec, sequences, out)


CKPT = "https://ml.jku.at/research/Bio-xLSTM/downloads/Prot-xLSTM/checkpoints/protxlstm_102M_60B"
protxlstm_image = (  # inference needs only the package's xLSTM code, not mamba_ssm
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install("torch==2.2.2", "transformers==4.44.2", "accelerate", "dacite", "omegaconf", "numpy<2", "pandas", "biopython", "rich", "wandb")
    .run_commands("git clone --depth 1 https://github.com/ml-jku/Prot-xLSTM.git /opt/prot-xlstm",  # its setup installs only the top package
                  f"mkdir -p /ckpt/protxlstm_102M_60B && cd /ckpt/protxlstm_102M_60B && wget -q {CKPT}/config.json {CKPT}/pytorch_model.bin")
    .env({"PYTHONPATH": "/opt/prot-xlstm"})
    .add_local_file("scripts/probe/protein_lm.py", "/work/protein_lm.py")
    .add_local_file("data/confirm_sequences.tsv", "/work/data/confirm_sequences.tsv")
    .add_local_file("data/small_family_sequences.tsv", "/work/data/small_family_sequences.tsv")
    .add_local_file("data/d_set_sequences.tsv", "/work/data/d_set_sequences.tsv")
)


@app.function(image=protxlstm_image, gpu=["L40S", "A100", "A100-80GB", "H100"], timeout=3 * 3600)
def run_protxlstm(spec: str, sequences: str, out: str) -> dict[str, str]:
    return run.local(spec, sequences, out)


@app.local_entrypoint()
def main(runs: str, sequences: str = "data/confirm_sequences.tsv", out: str = "results/plm_confirm"):
    specs = runs.split(",")
    Path(out).mkdir(parents=True, exist_ok=True)
    fn = run_progen3 if all(s.startswith("progen3") for s in specs) else run_protxlstm if all(s.startswith("protxlstm") for s in specs) else run
    for files in fn.map(specs, [sequences] * len(specs), [out] * len(specs)):
        for name, text in files.items():
            (Path(out) / name).write_text(text)
            print(name, text.count("\n"), "rows")
