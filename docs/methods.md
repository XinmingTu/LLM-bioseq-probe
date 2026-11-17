# Methods

## Protein test sets

- **Main set** (`data/confirm_sequences.tsv`, built by `scripts/data/04_confirm_sequences.py`):
  167 Pfam families named "domain/protein of unknown function" with no PDB structure, at most
  3 Europe PMC hits for the accession, at least 300 proteins, 50 families per size bin
  (300-1k, 1k-3k, 3k-10k, >10k proteins; the last bin ran out at 17). Per family, 4 unreviewed
  (TrEMBL) proteins from Bacteria or Archaea, 120-800 residues, standard residues only, not
  low-complexity, and alone in their UniRef90 cluster. 5 cut points per protein, uniformly at
  random inside the longest Pfam domain span. 668 proteins, 3,340 cuts.
- **Pilot set** (`data/sequences_pilot.tsv`): 37 families in a 2x2 of family size (many or
  few sequences) by literature (DUF with no structure, or 100+ PDB structures); 6 proteins per
  family, half added to UniProt before 2023 and half from June 2025. Used to choose the main
  design and for the well-known vs obscure comparison.
- **Why no post-cutoff sequences:** of about 3.8 million UniProtKB entries created between
  2026-07-01 and 2026-11-04, about 10 carry a sequence that UniParc first saw after
  2026-07-01 (`scripts/data/check_new_uniprot_sequences.sh`). UniParc ids are assigned in
  order; we bisected them against first-seen dates.

## Probes

- **Chat prompt** (`bioseq_probe.probe.prompt`): an instruction, then `>sequence` and the
  prefix in 60-character FASTA lines; the model returns the next 10 residues. Only the first
  is scored.
- **Shuffled control:** the same cut on the protein's residues shuffled with a fixed seed.
- **Best-guess prompt:** adds "Give your best guess even if you are uncertain. Do not explain
  or refuse." Used for models that declined the plain prompt (GPT-4o, DeepSeek, Qwen, Kimi,
  GLM, gpt-oss); reported separately.
- **Raw continuation** (`scripts/probe/protein_raw.py`): the FASTA text continued through
  Fireworks' completion endpoint, or as an assistant (Claude Sonnet 4.5) or model-turn
  (Gemini 2.5 Pro) prefill. Claude Opus 4.6 and later, GPT models and Gemini 3.5 and 4 accept
  neither.
- **Settings:** temperature 0 where accepted; thinking off where possible, else at its lowest
  level (`bioseq_probe.providers.MODELS`). One sample per question, no tools.
- **Refusals and non-answers** are excluded and counted. Claude Opus 5.5 left 0.1% unanswered;
  counting them as wrong moves its gap by under 1 point.

## DNA

220-nt windows (200 shown, 20 to predict) from hg38 and mm39 (UCSC .2bit, RepeatMasker
soft-masking; GENCODE basic canonical transcripts) and from the ENA coding sequences of the
main-set proteins. Main DNA set: 500 windows per region (human and mouse CDS, promoter 1 kb
upstream of the TSS, intron, intergenic with no gene within 5 kb; bacterial CDS), none
overlapping the pilot. Non-coding windows contain no repeat. The control shuffles each
window keeping its dinucleotide counts (`bioseq_probe.dna.dinucleotide_shuffle`).

## Statistics

- Score: real-minus-shuffled accuracy of the first residue (or nucleotide).
- Protein: per-family gap, 95% bootstrap CI over families (10,000 resamples), one-sided
  family-level sign-flip permutation test, Spearman correlation of the gap with log10 family
  size. DNA: the same over windows.
- Exact recall: cuts continued with all 10 residues (or 20 nucleotides) right, not present in
  the prefix, against the count expected if positions were independent at the model's own
  per-position accuracy.
- The main-set analysis plan was written before the main runs: `docs/preregistration.md`.
