"""Check that Prot-xLSTM is read correctly: next-residue accuracy on two well-known proteins under
different precisions and mLSTM backends, by prefix length, and with a homolog in context (the way it
was trained: <cls>-separated homologs, positions restarting at 0). About $0.50 of GPU time.

    PYTHONPATH=scripts/probe modal run scripts/probe/protxlstm_check.py
"""

import modal

from protein_lm_modal import protxlstm_image

app = modal.App("llm-bioseq-probe-protxlstm-check")
SEQS = {"ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        "hemoglobin beta": "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"}
MOUSE_HBB = "MVHLTDAEKAAVSCLWGKVNSDEVGGEALGRLLVVYPWTQRYFDSFGDLSSASAIMGNAKVKAHGKKVITAFNDGLNHLDSLKGTFASLSELHCDKLHVDPENFRLLGNMIVIVLGHHLGKDFTPAAQAAFQKVVAGVATALAHKYH"


@app.function(image=protxlstm_image.add_local_python_source("protein_lm_modal"), gpu=["L40S", "A100", "A100-80GB", "H100"], timeout=1800)
def check() -> list[str]:
    import math

    import torch
    from protxlstm.models.xlstm import xLSTMLMHeadModel
    from protxlstm.utils import AA_TO_ID, load_model

    aa = "ACDEFGHIKLMNPQRSTVWY"
    ids = [AA_TO_ID[a] for a in aa]
    out = []

    def enc(s):
        return [AA_TO_ID["<cls>"]] + [AA_TO_ID[a] for a in s]

    for dtype, backend in ((torch.bfloat16, "chunkwise_variable"), (torch.float32, "chunkwise_variable"), (torch.float32, "parallel")):
        model = load_model("/ckpt/protxlstm_102M_60B", model_class=xLSTMLMHeadModel, device="cuda", dtype=dtype,
                           mlstm_backend=backend, mlstm_chunksize=1024, mlstm_return_last_state=backend != "parallel").eval()
        for name, s in SEQS.items():
            x = torch.tensor([enc(s)], device="cuda")
            pos = torch.arange(x.shape[1], device="cuda").unsqueeze(0)
            with torch.no_grad():
                full = model(input_ids=x, position_ids=pos).logits[0].float()  # one pass, every position
            hits_full = [aa[int(full[c, ids].argmax())] == s[c] for c in range(10, len(s))]
            hits_prefix = []  # the way the probe reads it: one prefix per cut
            for c in range(10, len(s)):
                with torch.no_grad():
                    lg = model(input_ids=x[:, : c + 1], position_ids=pos[:, : c + 1]).logits[0, -1].float()
                hits_prefix.append(aa[int(lg[ids].argmax())] == s[c])
            lp = torch.log_softmax(full[:-1, ids], -1)
            nll = -sum(float(lp[c, aa.index(s[c])]) for c in range(len(s))) / len(s)
            half = len(hits_full) // 2
            out.append(f"{str(dtype):15s} {backend:18s} {name:16s} full-pass {sum(hits_full)}/{len(hits_full)} "
                       f"(first half {sum(hits_full[:half])}/{half}, second {sum(hits_full[half:])}/{len(hits_full) - half}); "
                       f"per-prefix {sum(hits_prefix)}/{len(hits_prefix)}; mean NLL {nll:.2f} (uniform 3.00), perplexity {math.exp(nll):.1f}")
        # positive control: mouse beta globin as one homolog in context, then human beta globin
        s, ctx = SEQS["hemoglobin beta"], enc(MOUSE_HBB)
        x = torch.tensor([ctx + enc(s)], device="cuda")
        pos = torch.tensor([list(range(len(ctx))) + list(range(len(s) + 1))], device="cuda")
        with torch.no_grad():
            full = model(input_ids=x, position_ids=pos).logits[0].float()[len(ctx):]
        hits = [aa[int(full[c, ids].argmax())] == s[c] for c in range(10, len(s))]
        lp = torch.log_softmax(full[:-1, ids], -1)
        nll = -sum(float(lp[c, aa.index(s[c])]) for c in range(len(s))) / len(s)
        out.append(f"{str(dtype):15s} {backend:18s} {'HBB + mouse ctx':16s} full-pass {sum(hits)}/{len(hits)}; "
                   f"mean NLL {nll:.2f}, perplexity {math.exp(nll):.1f}")
        del model
        torch.cuda.empty_cache()
    return out


@app.local_entrypoint()
def main():
    for line in check.remote():
        print(line)
