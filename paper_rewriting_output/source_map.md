# Source Map — GazeStateNet

Maps all evidence sources to claims in the manuscript.

| Source | Type | Path | Supports |
|--------|------|------|----------|
| GAFA paper | Published paper (CVPR 2022) | ./GAFA.pdf | Baseline; architecture reference; dataset description |
| UAGE paper | Published paper (ACCV 2024) | ./UAGE.pdf | Competitive baseline (20.5°); pose feature inspiration |
| GazeD paper | Published paper (3DV 2026) | ./GazeD.pdf | SOTA baseline (19.5°); gaze-point encoding inspiration |
| training logs | Experiment output | ./train_v6.log | V6 training metrics (val MAE, direction_loss) |
| test output | Experiment output | test_simple.py output | Test set MAE (21.48°) |
| per-scene output | Experiment output | eval_per_scene.py output | Per-scene breakdown |
| figures | Generated | ./figs/ | 8 paper figures |
| code | Source code | ./models/ | Implementation of all methods |
| ablation data | Collated results | v1-v7b experiments | Full ablation table |
| NaN analysis | Engineering log | Root cause analysis | Arccos gradient + rotation matrix fix |
| contribution doc | Analysis | ./confirmed_contribution.md | 5-contribution mapping |
| reviewer audit | Analysis | ./reviewer_audit.md | 7-objection analysis |
