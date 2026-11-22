# AUC Landmark Recognition

Recognizes AUC campus landmarks from phone photos using SIFT + RANSAC, KNN, SVM and MLP, and stitches panoramas. CSCE 4603 course project, Fall 2026.

## Layout

```
data/
  raw/<landmark>/        original photos (not in git, shared on Drive)
  processed/<landmark>/  resized copies made by data.resize_all()
  labels.csv             one row per photo; the only source of the split
src/
  config.py       all settings (landmark names, thresholds, seed)
  data.py         loading, resizing, train/test split        (teammate)
  features.py     Harris, SIFT, color histogram, PCA         (teammate)
  matching.py     ratio test, RANSAC, match figures          (me)
  recognition.py  method A: KNN on matches (me)
                  method B: SVM / MLP (teammate)
  panorama.py     stitching                                  (me)
  evaluate.py     accuracy, confusion matrices               (teammate)
  utils.py        show / save figures                        (both)
notebooks/        experiments and the final demo notebook
results/          saved figures and tables for the report
```

## Conventions

- Photo names: `<landmark>_<YYYY-MM-DD>_<index>.jpg`, e.g. `main_gate_2026-11-20_014.jpg`
- `split` in labels.csv is `train`, `test` or `pano`; test photos come from a different day than training photos.
- Every function that predicts returns landmark names, and all results go through `evaluate.evaluate()`.
- Numbers live in `config.py` only.
- Work on your own branch; merge to `main` only when it runs.

## Setup

```
pip install -r requirements.txt
```

Dataset: [link to the shared Drive folder]
