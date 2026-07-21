# Confirmed Motivation — GazeStateNet

> PaperSpine Gate 4: User-confirmed controlling motivation. This document locks the paper's narrative direction.

## Chosen Route: Efficiency-First + Ablation Depth (Route 1 + 3)

## One-Sentence Core Argument

> GazeStateNet demonstrates that five lightweight, purely observational gaze-state temporal features — computed from a frozen pretrained backbone at zero annotation cost — can improve GAFA 3D gaze estimation while using only 8% of the original trainable parameters, establishing a new efficiency-first paradigm for video-based behavioural feature engineering.

## What This Paper Is

- A demonstration that **lightweight computed features can compete with heavy learned ones** on the GAFA benchmark
- A **systematic ablation study** (7 configurations) documenting what temporal features work and why
- An **engineering contribution**: identifying and fixing the arccos gradient explosion NaN bug
- A **methodology contribution**: the "frozen backbone + computed features" paradigm

## What This Paper Is NOT

- A claim of SOTA accuracy (GazeD leads at 19.5°, UAGE at 20.5°)
- A proposal for a fundamentally new architecture (we reuse GAFA's LSTM design)
- A multi-person or social gaze model

## Target Audience

Computer vision researchers working on:
- Video-based gaze estimation
- Efficient deep learning (parameter reduction)
- Temporal feature engineering
- Surveillance and behavioural analysis

## Reviewer Takeaway

"These authors didn't chase SOTA with a bigger model. They found a clever way to squeeze more performance out of the existing GAFA pipeline with almost no extra cost, and they documented the whole process honestly — including what didn't work. The frozen-backbone + computed-features idea is transferable."

---

*Confirmed by user, 2026-07-22.*
