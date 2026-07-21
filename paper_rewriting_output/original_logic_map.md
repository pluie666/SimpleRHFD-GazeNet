# Original Logic Map — report.md

Logic flow analysis of the existing draft.

| § | Current Flow | Strength | Weakness |
|----|-------------|----------|----------|
| Abstract | Problem → method → numbers → significance | Complete; Nature-polished | — |
| §1 | Broad→specific→contributions→results | Clear 4-¶ structure; numbered contributions | — |
| §2.1-2.5 | UAGE→GazeD→GAFA→RHFD→Temporal | Honest comparison; credits all baselines | Section numbering bug (2.2 duplicate) |
| §3 | Architecture→features→fusion→loss→config | Formula-complete; code-verifiable | Architecture diagram is placeholder |
| §4 | Dataset→results→per-scene→ablation→NaN→curve | 7-config ablation; honest negatives | Some tables dense |
| §5 | Why RHFD→Why frozen→Limitations | Three clear arguments | — |
| §6 | Conclusion→Efficiency table | Paradigm statement; compelling comparison | — |

## Logic Flow Audit

Q: Does every claim in the Abstract appear in the body?
A: ✅ 21.48°, 0.82° frontal, 770K, NaN fix, frozen HBNet — all verified in §4.

Q: Does the reader know within ¶1 what this paper is about?
A: ✅ "Dynamic 3D gaze estimation from distant cameras..."

Q: Are the contributions falsifiable? Could someone prove us wrong?
A: ✅ All numbers are MAE measurements on a public benchmark. Our feature claims can be tested by anyone with the GAFA dataset.

Q: Is the paper honest about its limitations?
A: ✅ §5.3 lists 4 limitations including the marginal 5-vs-2 feature gain.
