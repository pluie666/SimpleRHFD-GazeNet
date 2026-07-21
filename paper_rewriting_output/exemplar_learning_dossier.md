# Exemplar Learning Dossier — GazeStateNet

## Exemplar Inventory (Pro Tier: 6 + 6 = 12 targets)

### Target Scene Examples (6 — CVPR/ICCV/ECCV conference papers)

| # | Paper | Year | Venue | Key Takeaway |
|---|-------|------|-------|--------------|
| 1 | GAFA (Nonaka et al.) | 2022 | CVPR | Two-stage pipeline; LSTM fusion of head+body; vMF uncertainty |
| 2 | Gaze360 (Kellnhofer et al.) | 2019 | ICCV | Large-scale unconstrained; 360° labels; LSTM temporal model |
| 3 | RT-GENE (Fischer et al.) | 2018 | ECCV | Real-time gaze; multi-modal input fusion |
| 4 | MPIIGaze (Zhang et al.) | 2015 | CVPR | Appearance-based; learning-by-synthesis; benchmark |
| 5 | iTracker (Krafka et al.) | 2016 | CVPR | Eye-tracking from mobile; multi-region CNN |
| 6 | Gaze360 (Recasens et al.) | 2017 | ICCV | Following gaze in video; social gaze modeling |

### Field/SOTA Examples (6 — Recent GAFA follow-ups)

| # | Paper | Year | Venue | Key Takeaway |
|---|-------|------|-------|--------------|
| 1 | UAGE (Lan et al.) | 2024 | ACCV | 4-branch body state + CVAE; domain adaptation; 20.5° on GAFA |
| 2 | GazeD (Catalini et al.) | 2026 | 3DV | Gaze as body joint + diffusion; scene context; 19.5° on GAFA |
| 3 | GazeD follow-up (IEEE) | 2024 | IEEE | Body pose + Transformer for gaze; 22.2° on GAFA |
| 4 | Eye Tracking for Everyone | 2016 | CVPR | Large-scale data collection methodology |
| 5 | Appearance-Based Gaze (Cheng et al.) | 2021 | arXiv | Deep gaze survey; benchmark comparison |
| 6 | End-to-End Video Eye-Tracking (Park et al.) | 2020 | ECCV | Video-based eye tracking pipeline |

## Structural Patterns Learned

From GAFA paper (CVPR 2022) — closest structural model:
- **Abstract**: Problem → method highlight → key numbers → significance
- **Introduction**: Motivation (5 daily scenes) → prior work gap → our approach → contributions (numbered) → results summary
- **Method**: Architecture diagram first → then component-by-component (HBNet → GazeModule) with formulas
- **Experiments**: Dataset table → main results table → ablation → per-scene → qualitative
- **Conclusion**: Recap contributions → future work

From GazeD paper (3DV 2026) — efficiency comparison model:
- Heavy emphasis on per-scene breakdown
- Training efficiency comparison table
- Oracle vs real-world performance gap discussion

## Rhetorical Patterns Learned

| Pattern | Example Source | How to Apply |
|---------|---------------|--------------|
| "We show that..." | GAFA intro | Open contribution paragraphs with active voice |
| "Our method achieves X, compared to Y for [baseline]" | All | Standard CVPR results language |
| "This suggests that..." | GazeD discussion | Hedge appropriately when interpreting results |
| "We also explored...found they did not improve..." | UAGE ablation | Honest negative result reporting |
| Front/back breakdown | GAFA Table 1 | Always report angular split |

## Language Patterns

- **Sentence length**: GAFA paper averages 18-25 words/sentence
- **Paragraph structure**: Claim → evidence → interpretation
- **Technical density**: ~2-3 formulas per method section page
- **Figure-to-text ratio**: ~1 figure per page
- **Citation density**: ~4-5 citations per page in related work
