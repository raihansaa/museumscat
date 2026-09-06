# examples/: synthetic stand-ins, so the pipeline runs with no competition data

Every file here is **fabricated**. The competition's per-row ground truth is not
redistributed (see the note in the top-level README), so these are hand-written rows that
have the same *shape* as the real data and exercise every branch of the code.

The place names are ordinary Danish geography; the `image_file` values are invented and
correspond to no real specimen.

| file | what it stands in for |
|---|---|
| `train.csv` | the 200 labelled rows: `image_file` + the two gold fields |
| `test.csv` | the 3,300 unlabelled rows: `image_file` only |
| `reads_a.csv` | one reader's output, as `run_reader.py` writes it (including the raw JSON) |
| `reads_b.csv` | a **second draw of the same reader**, for the self-disagreement cohort |

## What the rows are designed to exercise

- **`sample_0003`**: the reader answers `MISSING` while grading the field `clear`. That
  is a **self-contradiction**: the prompt requires an absent field to be graded `absent`.
  It is the cohort that was worth the most in the real campaign.
- **`sample_0005`**: a correct abstention: `MISSING` *and* graded `absent`. It must NOT
  be demoted, and it is why demoting `MISSING` blind fails.
- **`sample_0002`**: the two draws disagree substantively. The **self-disagreement**
  cohort.
- **`sample_0007`**: the two draws differ only cosmetically, and canonicalisation must
  collapse them. Demoting cosmetic disagreements measurably *cost* score.
- **`sample_0004`**: a multi-card answer with a pipe separator.
- **`sample_0006`**: German umlauts and an invented macron, for the `polish` folds.

## Try it

```bash
pip install -e ".[dev]"

# score a submission and print the ordering decomposition
python scripts/evaluate_submission.py \
    --submission examples/submission.csv --train examples/train.csv

# demote the two cohorts and confirm the text does not change
python scripts/apply_cohorts.py \
    --submission examples/submission.csv \
    --reads examples/reads_a.csv --reads-b examples/reads_b.csv \
    --field locality --disagreement --contradiction \
    --train examples/train.csv --out /tmp/demoted.csv

python scripts/build_submission.py --compare examples/submission.csv /tmp/demoted.csv
```

The last command should report both text columns `identical` and only the confidence
column differing. That is the property that made cohort demotion safe.
