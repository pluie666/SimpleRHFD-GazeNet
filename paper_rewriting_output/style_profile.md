# Style Profile — GazeStateNet

Derived from analysis of GAFA (CVPR 2022), UAGE (ACCV 2024), and GazeD (3DV 2026) papers.

## Structural Style

- **Section ordering**: Abstract → Introduction → Related Work → Method → Experiments → Discussion → Conclusion
- **Introduction pattern**: Broad problem statement → specific gap → our approach (numbered contributions) → key results
- **Method pattern**: Architecture diagram first → component-by-component with formulas → training details table
- **Experiment pattern**: Main results table → per-scene breakdown → ablation → qualitative → efficiency comparison

## Citation Style

- **Format**: Numeric [1], [2]... (IEEE/CVPR standard)
- **Density**: ~25-35 citations for a conference paper
- **Distribution**: Heavy in Related Work (15-20), light in Method (3-5), moderate in Discussion (5-10)

## Formula Style

- **Inline**: Simple expressions like $G_f$, $\kappa$, $\mathbf{h}_t$
- **Display**: Complex formulas like rotation matrix, loss functions, feature definitions
- **Notation**: Bold for vectors ($\mathbf{h}$), calligraphic for losses ($\mathcal{L}$), Roman for constants

## Table Style

- **Three-line tables**: Top rule, header rule, bottom rule; no vertical lines
- **Bold for best results**: Our method's best numbers in bold
- **Alignment**: Numbers centered, text left-aligned
- **Caption placement**: Above the table

## Figure Style

- **Vector preferred**: SVG/PDF for architecture diagrams
- **Consistent color scheme**: Soft pastels (no neon)
- **White background**: No dark mode figures
- **Caption below**: Descriptive, self-contained

## Language Style (Nature-Polished)

- **Sentence length**: 10-30 words average
- **Voice**: Active where natural ("We propose..."), passive for methods ("Features are computed...")
- **Hedging**: "may", "suggest", "appear to", "tend to" — not "proves" or "demonstrates definitively"
- **Spelling**: British English (behaviour, colour, generalisation, modelling)
- **No contractions**: "do not" not "don't"
- **No rhetorical questions**: "How can we improve..." → "We investigate how to improve..."
- **Paragraph unity**: One main idea per paragraph

## Target Venue Norms

| Norm | CVPR/ICCV Standard | Our Compliance |
|------|------|:---:|
| 8-page limit | Including references | ⚠️ Need to check |
| Double-blind | No author names | ⚠️ Need to anonymize |
| Supplementary | Unlimited pages | Available (code, logs) |
| Reproducibility | Code + data | ✅ GitHub + GAFA dataset |
