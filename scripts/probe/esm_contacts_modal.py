"""ESM-2 contact prediction on Modal GPUs -> results/contact_map/esm2_*.jsonl, one record per
protein with the L x L contact probabilities (fair-esm, return_contacts=True: the model's own
logistic regression on attention maps). A protein language model baseline for the LLMs'
contact maps (scripts/analysis/contact_map.py). Under $1.

Sequences come from results/contact_map/truth.json (scripts/probe/contact_map.py). ESM-2 was
trained on UniRef50 2021_04, before every recent protein here was public.

    modal run scripts/probe/esm_contacts_modal.py --models esm2_t33_650M_UR50D,esm2_t36_3B_UR50D
"""

import json
from pathlib import Path

import modal

app = modal.App("llm-bioseq-probe-esm-contacts")
MODELS = ("esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fair-esm==2.0.0", "torch==2.5.1", "numpy<2.3")
    .run_commands(*[f"python -c 'import esm; esm.pretrained.{m}()'" for m in MODELS])  # weights baked into the image
)


@app.function(image=image, gpu=["L40S", "A100", "A100-80GB", "H100"], timeout=3600)
def predict(model_name: str, sequences: dict[str, str]) -> dict[str, list[list[float]]]:
    import esm
    import torch

    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.eval().cuda()
    convert = alphabet.get_batch_converter()
    out = {}
    with torch.no_grad():
        for name, seq in sequences.items():
            _, _, tokens = convert([(name, seq)])
            probs = model(tokens.cuda(), repr_layers=[], return_contacts=True)["contacts"][0].float().cpu().numpy()
            assert probs.shape == (len(seq), len(seq)), (name, probs.shape)
            out[name] = probs.round(4).tolist()
    return out


@app.local_entrypoint()
def main(models: str = ",".join(MODELS), truth: str = "results/contact_map/truth.json", out: str = "results/contact_map"):
    t = json.loads(Path(truth).read_text())
    seqs = {name: v["sequence"] for name, v in t.items()}
    names = models.split(",")
    for model_name, probs in zip(names, predict.map(names, [seqs] * len(names))):
        path = Path(out) / f"{model_name.split('_UR50D')[0]}.jsonl"
        with path.open("w") as f:
            for name in seqs:
                f.write(json.dumps({"model": model_name, "protein": name, "pdb": t[name]["pdb"], "length": len(seqs[name]),
                                    "probs": probs[name]}) + "\n")
        print(path, len(probs), "proteins")
