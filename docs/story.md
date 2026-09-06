# The MuseumSCAT campaign, start to finish

This is the narrative companion to the write-up in [`../README.md`](../README.md). The write-up
says what the final system was and why it worked. This document says how we got there — every
phase, every tool, and the order in which we learned things, including the parts that went
nowhere.

The filenames in this document are the ones from the working repository as it stood at the
time. Only a subset of that code shipped here, reorganised into `museumscat/` and `scripts/`.
The rest were experiments that did not survive. See "What is in this repository" in the
write-up for the shipped layout.

Final: **private AURC 0.02300 / public 0.02203**, 14th of 126 teams, 85 submissions, across
roughly seven weeks.

Score journey:

```
0.10743  ->  0.03829  ->  0.03447  ->  0.03336  ->  0.02997  ->  0.02928
         ->  0.02852  ->  0.02626  ->  0.02606  ->  0.02474  ->  0.02263  ->  0.02203
```

---

## Phase 0 — The harness before the model

Nothing was read until the measurement apparatus existed. This turned out to be the single best
decision of the campaign, because almost every later result is a small difference between two
numbers, and small differences are only trustworthy if the numbers are.

What we built first:

- **`src/metric.py`** — a faithful reimplementation of the competition's NED + AURC scoring,
  including the multi-card permutation handling and the date normalisation. Every local decision
  was made against this, never against a proxy like exact-match accuracy.
- **`src/folds.py`** — grouped, stratified 5-fold CV over the 200 labelled rows. The grouping is
  by gold locality cluster and it matters enormously: the same place name recurs across many
  trays, so ungrouped folds put a row's near-duplicates in the training half and leak badly.
- **`src/preflight.py`** — fails loudly if the dataset on disk is not the dataset the caches were
  built against (SHA, row count, columns). Run before every harness.
- **`src/oof_harness.py`** — a frozen out-of-fold evaluation harness plus a per-image candidate
  archive, so that every read any model ever produced stayed on disk and could be re-scored later
  under a different rule without paying for inference again.
- **`src/check_submission.py`** — schema, row-count and id-alignment validation before any upload.
  It caught real defects more than once, and it is what let us prove at the end that our two
  final submissions carried byte-identical text and differed only in ranking.

The candidate archive deserves emphasis. Because every read was cached, most of the later
experiments cost nothing: re-scoring a cached reader under a new gate is free, and we ran dozens
of those. Cost discipline came from the archive, not from frugality about inference.

## Phase 1 — Cropping, local models, and the first submission (0.10743)

The images are 8192×5464 and the labels occupy a small band. Sending the whole photo to a model
downscales the handwriting into illegibility, so the first component was a cropper.

- **`src/crop_labels.py`** — crops the label band from a registered tray image.
- **`src/crop_tag.py`** — the version we actually used: detects the pale label rectangles by
  brightness and crops tightly to them.

Reading was local at first, through **Ollama**, using **`src/probe_vlm.py`** — a JSON-emitting
prompt over the crops, temperature 0. The first working reader was **qwen2.5-VL-7B**.

Around it we assembled the pieces that stayed for the rest of the campaign:

- **`src/polish.py`** — deterministic, metric-justified string cleanup. Every rule was validated
  against the ground truth for the rows it *changes* as well as the rows it *fixes*.
- **`src/make_submission.py`** — assembles the exact competition format.
- **`src/conf_ranker.py`** — a gradient-boosted regressor (scikit-learn) that predicts a row's
  NED. Its prediction *is* the confidence column. This is the only trained component in the
  system and it was never trained on images, only on features of the reads.

First shipped score: **0.10743**.

## Phase 2 — Cloud readers, and the ensemble (→ 0.03829)

Local 7B models were the bottleneck. We wired **`src/cloud_vlm.py`** — an OpenAI-dialect client
against **OpenRouter**, deliberately reusing `probe_vlm`'s image encoder so that cloud reads and
local reads were directly comparable.

The first cloud reader was **qwen3-VL-32B**, at a measured $0.00026 per image — $0.86 for the
full 3,300. That number mattered: our pre-measurement estimate had been one to two orders of
magnitude too high, and correcting it changed which experiments were affordable.

The gate we ran on it set the pattern for everything after. Three variants were measured:

| variant | what changed | result |
|---|---|---|
| **B** | locality text only, ordering frozen, nothing retrained | **best** |
| C | B, plus the ranker retrained on the new reader | worse |
| E | B, plus dates also swapped | worse — *even though the 32B reads dates better* |

Variant E is the instructive one. The 32B genuinely transcribed dates more accurately
(NED 0.087 → 0.072) and swapping them still hurt, because the date ranker's ordering was
calibrated against the 7B's error pattern. **Better text you cannot rank is worth nothing.** That
sentence governed the rest of the campaign.

Two more readers were added — **GPT-4o** and **Claude Sonnet 5** — giving a four-reader ensemble
combined by per-row majority vote with grammar gating. This is where the **agreement stratum**
(4–0, 3–1, 2–2, 2–1–1, 1–1–1–1) enters, initially just as a diagnostic.

## Phase 3 — Substitution beats voting (→ 0.03336)

Partitioning the labelled rows by agreement stratum showed that two disagreement strata held 106
of 200 rows and **83.9% of the locality AURC loss**, while unanimous reads took 27.5% of the
metric's weight and produced 5.9% of its loss.

The obvious move was a fifth, stronger reader. The non-obvious part was *how* to use it: adding
**claude-opus-5** as an extra **vote** cannot overturn a string that already holds a plurality, so
it structurally cannot touch the 2–1–1 stratum — which the measurement said was the single largest
prize. Replacing the text directly does reach it.

Frozen-rank substitution — swap the string, leave the confidence column untouched — beat a five-way
vote by roughly 5×. Score **0.03336**.

## Phase 4 — Gemini, and the reader question closing (→ 0.02997 → 0.02928)

**gemini-3.7-flash** turned out to read better than Opus-5 at about a ninth of the cost. Measured
head to head on 63 hard and 20 safe labelled rows:

```
gemini-3.7-flash   removes 8.68 NED   19/63 exact   breaks 0/20 safe    <- winner
gemini-3.6-flash   removes 6.48       18/63         breaks 1/20
claude-opus-5      removes 3.73        8/63         breaks 0/20
```

Cost $0.0028/row; a full 2,281-row pass was $5.74 against $44.55 for the same pass on Opus.

Operational details that were necessary rather than incidental:

- **`src/gemini_filter.py`** — Gemini's one systematic defect is inventing macrons
  (`Vamdrūp`, `Fālster`), about 2% of strings, each turning a correct read wrong. The fix is a
  named-character table, *not* an NFD strip, because a blanket strip destroys Danish `Å`
  (Århus → Arhus).
- **`harvest_reads.sh`** — the direct Gemini API free tier allows 20 requests per day *per model*,
  and quotas are per-model, so rotating a validated pool of six Gemini variants yields around 120
  free validated reads a day. Resumable, and it retires a model on a per-day 429.
- **Never put `-lite` models in the pool.** `gemini-3.5-flash-lite` returned a collector's name as
  a locality and mangled Danish on blank labels.

Substitution restricted to the strata that cleared a bootstrap bar moved us to **0.02997**; a
coordinate-field fix to **0.02928**.

We then closed the reader question by re-scoring cached reads for the whole GPT-5.6 class:

| reader | polished NED | exact |
|---|---|---|
| gpt-5.6-luna | 0.2747 | 56.9% |
| claude-opus-5 | 0.0665 | 79.2% |
| **gemini-3.7-flash** | **0.0508** | **80.0%** |
| gpt-5.6-sol | 0.3824 | 26.8% |
| gpt-5.6-terra (priciest) | 0.4777 | 9.8% |

After this, "try a better reader" stopped being an open lever. `gemini-3.1-pro` was never
reachable — it exists only on the free-tier direct key, whose quota we had exhausted, and
OpenRouter does not carry it.

## Phase 5 — The ordering era (→ 0.02203)

By this point the text was as good as we could make it, and the decomposition said the remaining
loss was mostly ordering. Two facts had accumulated:

- **Wholesale confidence re-ranking never worked — 0 for 8.** Both a trained re-ranker
  (`experiments/conf_v2.py`) and an untrained heuristic one (`conf_v3.py`) cost about +0.011 on
  their field, including variants that reached `P(better) = 1.000` on the labelled set.
- **Order-preserving cohort demotion always worked — 8 for 8.**

So the operator we shipped is narrow: identify a rule-defined cohort, push it behind every row
outside it, leave all other relative order exactly as it was, and change no text.

The cohorts, in the order they shipped:

| submission | cohort | rows | public AURC |
|---|---|---|---|
| `legDEMOTE` | legibility head — a 100%-error cohort | — | 0.02852 |
| `bothDEMOTE` | both-field extension | — | 0.02626 |
| `gDEMOTE` | third cohort; the well runs dry here | — | 0.02606 |
| `scLOC` | locality two-draw self-disagreement | 424 | 0.02474 |
| `scMISS` | locality self-contradiction tier | 100 | 0.02263 |
| `scMISS2` | + date false-`MISSING` union | 108 | **0.02203** |

Two things were learned here that we would carry to any other selective-prediction task.

**Measure a cohort's error rate before shipping it.** A 100%-error cohort pays; cohorts at 25%,
15% and 11% all cost score. `legDEMOTE` is also the case where our own gate was wrong — the n=200
bootstrap rejected it at `P = 0.883`, we shipped it anyway on mechanism, and it became the first
ordering win of the campaign.

**Rank by expected error magnitude, not error probability.** AURC charges a wrong row by *how*
wrong it is. The worst possible row is therefore a false `MISSING`: the system emits nothing where
the gold record has text, scoring NED 1.0 — and our confidence probe was mapping `absent`
legibility to 0.85, the second-highest band. A maximal error carried at high rank. The signal that
found them was a **self-contradiction in the model's own JSON**: a row returning `MISSING` while
grading that same field `clear` or `partial` has violated its own instructions. It fired 8 times
on the labelled set and was wrong all 8 times, with a real gold value each time.

The break-even arithmetic is the transferable part. Demoting `MISSING` blind fails, because
correct abstentions outnumber false ones 30:6 and 59:5 — but those same counts say a demoted row
only needs **29% precision (locality) / 9% (date)**, not the ~98% we had assumed a verifier
needed. We had abandoned a perfectly good instrument earlier for missing a bar that did not apply.

That session was worth −0.00403, the largest of the campaign, and it moved us from rank 14 to 12
on the public board.

## Phase 6 — Everything that did not work

Roughly two thirds of the effort went here. Listed because the negative results are the reusable
part, and because several are ideas any team would try.

**Training our own model.** `src/synth.py` (a synthetic Danish label generator with handwriting
fonts and degradation), `src/build_sft_dataset.py` and a LoRA fine-tune via **unsloth**. Never
beat the hosted readers on the fields that mattered.

**Target-blind structure.** `src/embeddings.py` and `src/graphs.py` built writer, stock and
structural graphs over the crops, on the theory that specimens filed together share a collector
and a locality. A transductive duplicate-card text transfer broke 44 rows and fixed 7, at 1.3%
coverage.

**Token-probability confidence.** `src/logprob_features.py` extracted per-field token
probabilities from Ollama logprobs. Later, a full visual-grounding line: the score is genuinely
visual (98.5% / 99.0% against wrong-image controls) but does not separate correct from incorrect —
AUROC sat inside the permutation null. The one variant with real signal (p = 0.001) still cost
+0.0038 / +0.0018 on the leaderboard. **AUROC is not AURC.**

**A forced-choice visual verifier.** Built and run for $0.47. Certified precision 81% (locality) /
94.9% (date). The mechanism is worth knowing: *the judge was the reader.* Asked to adjudicate its
own output, gemini rejected its own read only 38.5% of the time. A verifier must come from a
different model family.

**Few-shot exemplar images.** Showing the model correctly-transcribed example labels alongside the
query **hurt** by +0.0018 — it pattern-matches against the exemplars instead of reading. Stopped at
the labelled gate for $0.55; no submission slot spent.

**Image enhancement and alternative OCR.** Sharpening, contrast enhancement, RapidOCR. All null.
The error taxonomy explains why: 49% of locality loss is multi-card *segment selection*, not
character recognition. Pipe separators are canonicalised by the metric — joining segments changes
NED by 0.0003 — so the loss is in which physical cards get transcribed at all.

**A published-paper audit.** We audited arXiv [2608.22366](https://arxiv.org/abs/2608.22366)
(Arabic VLM-OCR) against our data: OCR-conditioned correction, output-shape cohorts, a
hallucination flag and image sharpening. All four failed their gates here. The audit did produce
one genuinely valuable finding, below.

**Snapping rare spellings to frequent ones.** This is a verbatim task and the historical spellings
are frequently correct. Vocabulary frequency stayed a *confidence* signal and was never allowed to
edit text.

## Phase 7 — Two measurement traps

**gemini-3.7-flash is not deterministic at temperature 0.** Only 34 of 40 localities reproduced on
a byte-identical re-read. A single-run n=200 reader A/B therefore carries a noise floor of about
±0.002 — the same size as the effects we were shipping. We found this only because one experiment
included a control arm that changed nothing. **Always buy the control.**

In hindsight this is also the largest lever we left on the table. We recorded the variance as
*noise to defend against*. It is equally a *signal to exploit*: our two-draw self-disagreement
cohort paid −0.00132, and we never pushed past K=2. Two draws let you detect that a read is
unstable; three or more let you resolve which one to keep.

**200 labelled rows is below what validating a ranking lever needs.** Our labelled harness
understated provable fixes by up to 15× and overstated others by 2×. Near the end the measurement
error exceeded the effects we were testing. Structural changes transferred to the leaderboard;
parameter tuning did not.

## Phase 8 — Close-out

Two final submissions are scored and the better private result counts, so slot 2 is free and the
objective is to maximise `E[max(A, B)]` — which is **not** achieved by picking the two highest
expected scores. The runner-up by score fails in exactly the world the leader fails. We picked the
**decorrelated** candidate instead: `scMISS2` (0.02203) and `gDEMOTE` (0.02606), which carry
byte-identical text and differ only in ranking, hedging the newest and least-replicated tier of
displacement.

Verified before ticking: 3,300/3,300 rows, ids aligned, `verbatimDate` and `verbatimLocality`
byte-identical between the two files, confidences differing on 46.4% (date) and 97.8% (locality)
of rows.

---

## Inventory — everything we used

**Readers (hosted).** gemini-3.7-flash (primary), gemini-3.6-flash, gemini-3.5-flash,
gemini-3-flash-preview, gemini-flash-latest, gemini-2.5-flash, claude-opus-5, claude-sonnet-5,
GPT-4o, qwen3-VL-32B, and gpt-5.6 luna / sol / terra (all three tested, all three rejected).

**Readers (local).** qwen2.5-VL-7B and qwen3-VL-8B via Ollama.

**Services.** OpenRouter (OpenAI-dialect, most cloud reads), the direct Gemini API (free tier,
20 requests/day/model, pool-rotated), Ollama (local inference), Kaggle (submissions).

**Libraries.** pandas, numpy, scikit-learn (the gradient-boosted confidence ranker), OpenCV and
Pillow (cropping and image handling), unsloth (the LoRA line, abandoned).

**What we never used.** No model was trained on the competition images. No external gazetteer or
place-name database was consulted. No handwriting-recognition model was fine-tuned. No test-time
sampling above K=2.
