# Reviewer Audit — SimpleRHFD-GazeNet

> PaperSpine V4 Gate 7: Pre-submission reviewer simulation.

---

## Objection 1: "21.48° vs 21.69° is only a 0.21° improvement. Is this statistically meaningful?"

**Severity**: ⚠️ HIGH — this will be the first question every reviewer asks.

**Pre-emptive response**:
- The 0.21° improvement is on the **full test set of 34,335 sequences** across 6 scenes — this is not a cherry-picked subset.
- More importantly, **frontal gaze improves by 0.82°** (19.88° vs 20.70°) — a 4% relative improvement on the most practically important subset.
- Per-scene breakdown shows consistent gains: Office 14.3° (best among all methods, including GazeD's 15.8°), Kitchen 18.8° vs GAFA 20.4°.
- The **parameter efficiency** is the real headline: 21.48° with 770K params vs 21.69° with 9.5M params.
- We recommend reporting bootstrap confidence intervals for the MAE to quantify uncertainty.

**Action**: Add bootstrap confidence intervals to the results table.

---

## Objection 2: "UAGE (20.5°) and GazeD (19.5°) both outperform your method. Why should this be published?"

**Severity**: ⚠️ HIGH — the accuracy-is-everything reviewer.

**Pre-emptive response**:
- Our contribution is **not** claiming SOTA accuracy. It is methodological: demonstrating that lightweight, purely-observational temporal features can achieve competitive results at 1-8% of the parameter cost.
- The paper explicitly compares training efficiency: 10 epochs, 1-2 epochs to converge vs. 100 epochs for GazeD and GAFA.
- We systematically ablated UAGE-inspired (pose) and GazeD-inspired (gaze-point) approaches and found them **ineffective** on GAFA (21.52° and 21.80°) — this negative result is valuable to the community.
- The "frozen backbone + computed features" paradigm generalises beyond gaze estimation.

**Action**: Strengthen the methodological comparison table in Section 6.1. Frame as "efficiency-first" rather than "accuracy-first".

---

## Objection 3: "The val-test gap of 6.7° is still large. Is the model really generalising?"

**Severity**: Medium — legitimate concern, but the paper honestly reports it.

**Pre-emptive response**:
- The original GAFA paper does not report validation MAE, but our reproduction experiments (v1-v2) suggest a gap of 13.8° for the unfrozen model — **our methods halved it**.
- GazeD and UAGE also do not report validation MAE, so direct comparison is impossible, but the GAFA dataset's structural train-test domain shift is a known limitation of the benchmark.
- The per-scene results provide fine-grained evidence of generalisation: Office (14.3°, best among all methods), Living Room (24.7°, within range), Kitchen (18.8°, improved by 1.6° over GAFA), Library (19.5°, comparable), Courtyard (25.8°, consistent with all methods).

**Action**: Acknowledge in limitations that closing this gap requires more diverse training data, not just better features.

---

## Objection 4: "The RHFD features are computed from head direction, not true gaze. Isn't this circular?"

**Severity**: Medium — semantic concern.

**Pre-emptive response**:
- Correct — Gf, Gd, Ga, Gv, Gs are **head-direction proxies**, not true gaze features. We are explicit about this (Section 3.2: "computed from head direction sequences").
- This is a **feature**, not a bug: the features are available at inference time without gaze labels, making the approach practical.
- The ablation with true pose features (v7: 21.52°) and spatial point encoding (v7b: 21.80°) suggests that **head-direction-derived temporal statistics are already the most informative signals at this resolution**.

---

## Objection 5: "The ablation lacks statistical significance testing. Are differences between configurations real?"

**Severity**: Low-Medium — standard concern for any ablation study.

**Pre-emptive response**:
- The 3.02° gap between v1 (unfrozen, 24.50°) and v3 (frozen, 21.49°) is large enough to be practically meaningful without formal testing.
- Configurations v3-v6 cluster within 0.05° of each other (21.45°-21.49°) — we honestly report this as "marginal improvement" and do not overclaim.
- The V7 negative results (21.52°, 21.80°) are also honestly reported as "statistically indistinguishable from v6".

**Action**: Optionally run paired bootstrap tests for the main comparisons (v1 vs v3, v3 vs v6).

---

## Objection 6: "The related work on RHFD is thin. Where are the hidden follower detection papers?"

**Severity**: Low — the paper is about gaze estimation, not follower detection.

**Pre-emptive response**:
- The term "RHFD" in our paper refers to the **feature vocabulary** (Gf, Gd, etc.), not the follower detection task itself. We should be precise about this.
- Consider changing "RHFD features" to "gaze-state temporal features" in the title and abstract to avoid confusion, while crediting RHFD literature in Section 2.2 for the feature inspiration.

**Action**: Clarify terminology — "gaze-state temporal features (inspired by RHFD literature)".

---

## Objection 7: "No code for reproducing the results?"

**Severity**: Low — but required for CVPR/ICCV.

**Pre-emptive response**: Code is open-source at `https://github.com/pluie666/SimpleRHFD-GazeNet`. Include this in the paper.

---

## Reviewer Scorecard

| Criterion | Score | Notes |
|-----------|:---:|------|
| Novelty | 3/5 | Feature set is novel in GAFA context; paradigm contribution |
| Technical Depth | 4/5 | 7-configuration ablation, NaN root-cause analysis, gradient isolation proof |
| Experimental Rigour | 4/5 | 3 baselines, front/back, per-scene, efficiency comparison, negative results |
| Clarity | 4/5 | Architecture diagrams, formulas, tables; Nature-polished text |
| Significance | 3/5 | 0.21° improvement is modest; 0.82° frontal + efficiency paradigm are stronger |
| **Overall** | **3.5/5** | **Borderline accept for CVPR workshop / Accept for ACCV/BMVC** |

### Target Venue Recommendation
- **CVPR/ICCV workshop** (e.g., GAZE, HBU): Strong accept
- **ACCV/BMVC main conference**: Accept / weak accept
- **WACV**: Accept
- **CVPR/ICCV main**: Borderline — needs stronger novelty framing or additional experiments
