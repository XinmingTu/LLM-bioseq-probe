# Findings

**Claude Fable 5.2** (run through OpenRouter; Anthropic's API serves it only to workspaces that retain
data): +45.1 points (95% CI 42.5 to 47.6), 54.1% vs 9.0%, positive in 98% of families, +45.9 in the
families with no papers. On the main set it is flat across family size (rho 0.00) because it is
already at its ceiling there: on 90 smaller families (20 to 300 proteins) it rises from +19.5 to
+35.5, then levels off near +45 (rho 0.38 over all 257 families). It matches Opus 5.5 and ProtGPT3-10B on
which families are easy (Spearman 0.63 and 0.57) and reproduces on a repeat of 400 questions (57.5%
vs 57.5% real). On the pilot it scores +70.3 on well-known and +53.3 on obscure proteins, the highest on both.
It thinks on 9% of questions; without thinking it scores 54.7% vs 8.8%.

**Claude Fable 5.1**: +14.9 (12.6 to 17.2), never thinks, rho 0.48 over 257 families.
**Claude Fable 5**: Anthropic's filter blocked 44% of its questions; on the rest +13.9 (11.6 with
blocked questions counted as wrong, 14.6 on cuts answered on both real and shuffled).
**GPT-6.1 Sol**: +0.6 (-0.7 to 2.0).

Family size over all 257 families (gap in points, bins 20-60 / 60-150 / 150-300 / 300-1k / 1k-3k /
3k-10k / >10k): Opus 5.5 +7.9 / +7.4 / +14.4 / +21.2 / +26.0 / +31.7 / +36.7 (rho 0.50);
ProtGPT3-10B +3.2 / +6.5 / +10.7 / +23.9 / +25.3 / +30.6 / +37.1 (rho 0.46); Fable 5.2 +19.5 / +24.0 /
+35.5 / +44.1 / +46.0 / +46.4 / +41.5 (rho 0.38).

Run through each provider's API in November 2026. Figures: `figures/fig0_method` to `fig5_family_size`.

## Claude Opus 5.5 was trained on protein sequence data

On 668 bacterial and archaeal proteins from 167 unknown-function families (no papers to speak
of, no relative at 90% identity), Claude Opus 5.5 gets the next amino acid right 34.9% of the
time, against 7.5% on the same residues shuffled and 5% for a blind guess: a gap of 27.4
points (95% CI 24.7 to 30.0, sign-flip permutation p < 0.0001).

- The gap appears in 94% of families (median 25 points); the 10 largest families give 14% of it.
- It grows with family size (Spearman 0.27, p = 0.0002): 22 points for families of 300-1k
  proteins, 37 for families over 10k. In the 128 families with zero Europe PMC hits it is
  26.8 points and still grows with size (rho 0.28, p = 0.001); it does not track paper counts
  (rho 0.05).
- It is not reasoning at test time: on the 55% of questions where Opus 5.5 wrote 15 tokens or
  fewer, it scores 35.6% vs 7.3%. GPT-6 Astra thinks for 200+ tokens a question and reaches
  about 10%.
- It is not the prompt: asking Opus 5.5 to copy the sequence and continue gives the same result.
- It is not recall of UniProt pages: entries on uniprot.org before 2023 and entries added from
  June 2025 score alike (37.2 vs 37.6 points, pilot).
- Exact recall of these obscure proteins is weak: 12 cuts continued 10/10 where 0.57 are
  expected from per-position accuracy; re-asked for 200 residues, Opus 5.5 rarely goes past 10.
  It learned the families' statistics more than each protein's letters.

## Within Claude, the signal is recent

Claude Opus 4.5: +1.9 points. Opus 4.6: +0.9. Opus 4.7: +8.9 (6.9 to 11.1). Opus 4.8: +7.7
(5.5 to 9.8). Opus 5: +9.7 (7.3 to 12.2). Opus 5.5: +27.4. Opus 4.7, 4.8 and 5 also grow with
family size (rho 0.37, 0.31 and 0.29), so the data was there from Opus 4.7 on and much stronger
in Opus 5.5.

## Opus 5.5 matches the largest open protein language model

Given only the residues before the cut, like the LLMs: ProtGPT3-10B (2026, left to right) +27.7
(24.2 to 31.1), against +27.4 for Opus 5.5. Smaller models score lower: ESM C 600M +13.0 (10.7 to
15.2), ProtGPT3-1.3B +11.6 (8.9 to 14.4), ProtGPT2 +10.2 (8.3 to 12.2), ESM3-open 1.4B +8.5 (6.4 to
10.7), ESM-2 650M +7.0 (5.2 to 8.9). ESM-2 given the residues on both sides of the cut reaches
+34.6 (31.8 to 37.3). ProtGPT3-10B is more accurate on real proteins (38.4% vs 34.9%) and on shuffled
ones (10.7% vs 7.5%); the gap, what each learned about residue order, is the same. ProtGPT3-10B also
follows Opus 5.5's family-size curve bin for bin (23.9, 25.3, 30.6, 37.1 against 21.2, 26.0, 31.7,
36.7), and on the pilot set it keeps most of its accuracy from well-known to obscure proteins (+51.4
to +38.2), like Opus 5.5 (+51.9 to +33.3) and unlike Gemini 4 Carbon (+47.6 to +5.3). Sanity checks
and method: `scripts/probe/protein_lm.py`.

## No other model shows it

GPT-5.5 (+2.1), GPT-6 Astra (+2.9), Gemini 4 Carbon (+0.9) through the chat API; Gemini 2.5
Pro (+1.1) and seven open-weight models (DeepSeek V4.1 Flash, Qwen 3.8, Kimi K3.1 Pro, GLM-5.3,
MiniMax M3, gpt-oss-120b, Nemotron 3 Ultra: -0.0 to +1.9) as raw next-token continuation of
the FASTA text. Absence of a signal does not prove absence of training data.

## Famous proteins prove little

On well-studied vertebrate proteins most models do well (Gemini 4 Carbon +47.6 points, Opus
4.8 to 5.5 +44 to +52), and several recite hundreds of residues letter for letter (Opus 5
longest 325, Opus 5.5 314, Gemini 4 Carbon 201). These sequences fill papers and web pages.
GPT models are the exception: GPT-5.5's longest exact run on these proteins is 2 residues.

## DNA: coding sequence only

On 500 repeat-free windows per region, Opus 5.5 predicts human coding sequence (+30.5, 25
exact 20-nucleotide continuations) and mouse coding sequence (+28.5, 5 exact), and nothing
clear in promoters, introns or intergenic DNA (-0 to +5). The DNA of the bacterial genes
behind the proteins in the main set gives +6.0 (p = 0.026), against +27.4 as protein: the
training data held protein sequences, not the genomes they come from.

## Safety behaviour seen along the way

- Gemini 4 Carbon blocks about 75% of DNA prompts before generation (`block_reason OTHER`).
- Claude Sonnet 4.5 refused 97% of protein prefill requests.
- OpenAI rejected 23 GPT-6 requests as "flagged for possible biological" misuse.
- GPT-4o, DeepSeek V4.1, gpt-oss and others often declined to guess a sequence ("I can't
  reliably predict…") until asked for a best guess.

## What this does not show

Whether the sequences came from a deliberate sequence corpus or crawled database pages; in
which training phase they entered; whether any lab trained on protein-model outputs; that the
other models never saw sequences.

## Related work

[PG-LLM](https://www.proteingymllm.com/) (Arora, Chen, Du, Marks and Church, 2026) ranks protein
variants with general-purpose LLMs; Claude Opus 5 approached ESM-2, with accuracy that barely
changes with alignment depth. A [post by Atharv Jayprakash](https://x.com/atharvjay872/status/2121477118450340342)
found GPT-6 Astra ranking SOD1 variants about as well as ESM-2. Variant ranking can draw on
biology and the literature; the test here removes that route, and GPT-6 shows almost no
sequence knowledge on it.

## Replication and parsing

- Replies are read with `bioseq_probe.probe.parse` (the last stated sequence; words, lists and
  thinking tags skipped; a restated prefix removed). Re-reading every archived reply with it
  instead of the earlier last-line reader moves no main-set gap by more than 0.1 point and no
  family-size bin by more than 0.4; Opus 5.5's longest recitation becomes 314 (was 305).
- ProGen3-3B was also run independently on one H200: +32.3 (29.1 to 35.4) against +32.5 on
  Modal, with the same top residue on 97.7% of questions. The rest is run-to-run variation in
  ProGen3's RMSNorm kernel, which picks a launch configuration per process.
