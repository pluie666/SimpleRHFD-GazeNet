# Research Dossier — GazeStateNet

## Venue: Computer Vision Conference (CVPR/ICCV/ECCV Style)

### Venue Requirements

| Requirement | Status | Notes |
|-------------|:---:|------|
| 8-page limit (excluding references) | ⚠️ | Current draft ~6 pages; room for expansion |
| Double-blind submission | ⚠️ | Need to anonymize if targeting double-blind venue |
| Novel contribution | ✅ | 5 RHFD features + gradient isolation + frozen HBNet |
| SOTA comparison | ✅ | GAFA (21.69°), UAGE (20.5°), GazeD (19.5°) |
| Ablation study | ✅ | 7 configurations (v1-v7b) |
| Per-scene breakdown | ✅ | Office/LR/Kitchen/Library/Courtyard |
| Code release | ✅ | github.com/pluie666/GazeStateNet |

### Review Criteria (CVPR Standards)

1. **Novelty (3/5)**: RHFD features are new in GAFA context; gradient isolation fix is practical; frozen-backbone paradigm is a methodology contribution
2. **Technical quality (4/5)**: 7-configuration ablation, NaN root-cause analysis, honest negative results
3. **Experimental rigor (4/5)**: 3 baselines, front/back, per-scene, efficiency comparison
4. **Clarity (4/5)**: Well-structured with formulas, tables, and 8 figures
5. **Significance (3/5)**: 0.21° improvement modest; 0.82° frontal + paradigm more compelling

### Accepted Paper Patterns (GAZE / HBU workshops, WACV, ACCV)

Common patterns in accepted papers at this venue tier:
1. **Clear gap statement in first paragraph**: What prior work misses → what we add
2. **Honest comparison**: Not claiming SOTA when we're not; positioning as efficiency-first
3. **Negative results**: Reporting that UAGE/GazeD-inspired features didn't help → valued by reviewers
4. **Generalizable insight**: "Freeze backbone + computed temporal features" → applies beyond gaze

### Constraints for This Paper

| Constraint | Impact |
|------------|--------|
| UAGE (20.5°) and GazeD (19.5°) both outperform us | Cannot claim SOTA; must frame as efficiency/paradigm contribution |
| 0.21° improvement over GAFA | Modest; must emphasize 0.82° frontal + parameter reduction |
| No face-crop available | Back-facing gaze inherently limited |
| Single-person scenes only | Cannot claim multi-person generalization |
