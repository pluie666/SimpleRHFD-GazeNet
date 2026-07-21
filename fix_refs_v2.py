"""Fix references in report.md: remove 11 irrelevant refs, renumber to 1-17, add RHFD."""
with open('report.md', 'r', encoding='utf-8') as f:
    content = f.read()

# In-text remapping (old->new) — apply from high to low
remap = {28: 17, 27: 16, 26: 15, 18: 14, 17: 13, 16: 12, 13: 11}
for old, new in sorted(remap.items(), reverse=True):
    content = content.replace(f'[{old}]', f'[{new}]')

# Clean reference list
new_refs = """## References

[1] S. Nonaka, S. Nobuhara, and K. Nishino. Dynamic 3D Gaze from Afar: Deep Gaze Estimation from Temporal Eye-Head-Body Coordination. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 2192-2201, 2022.

[2] N. I. Fisher, T. Lewis, and B. J. J. Embleton. *Statistical Analysis of Spherical Data*. Cambridge University Press, 1987.

[3] I. Loshchilov and F. Hutter. Decoupled Weight Decay Regularization. In *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.

[4] M. Tan and Q. V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In *Proc. Int. Conf. Machine Learning (ICML)*, pp. 6105-6114, 2019.

[5] S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. *Neural Computation*, 9(8):1735-1780, 1997.

[6] X. Zhang, Y. Sugano, M. Fritz, and A. Bulling. Appearance-Based Gaze Estimation in the Wild. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 4511-4520, 2015.

[7] P. Kellnhofer, A. Recasens, S. Stent, W. Matusik, and A. Torralba. Gaze360: Physically Unconstrained Gaze Estimation in the Wild. In *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, pp. 6912-6921, 2019.

[8] T. Fischer, H. J. Chang, and Y. Demiris. RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments. In *Proc. European Conf. Computer Vision (ECCV)*, pp. 334-352, 2018.

[9] Y. Sugano, Y. Matsushita, and Y. Sato. Learning-by-Synthesis for Appearance-Based 3D Gaze Estimation. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 1821-1828, 2014.

[10] K. Krafka, A. Khosla, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik, and A. Torralba. Eye Tracking for Everyone. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 2176-2184, 2016.

[11] I. Loshchilov and F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. In *Proc. Int. Conf. Learning Representations (ICLR)*, 2017.

[12] A. Paszke, S. Gross, F. Massa, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In *NeurIPS*, pp. 8024-8035, 2019.

[13] W. Falcon et al. PyTorch Lightning. 2019. https://github.com/Lightning-AI/lightning.

[14] D. P. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. In *Proc. Int. Conf. Learning Representations (ICLR)*, 2015.

[15] E. Lan, Z. Hu, and J. Liu. UAGE: A Supervised Contrastive Method for Unconstrained Adaptive Gaze Estimation. In *Proc. Asian Conf. Computer Vision (ACCV)*, 2024.

[16] R. Catalini et al. GazeD: Context-Aware Diffusion for Accurate 3D Gaze Estimation. In *Proc. Int. Conf. 3D Vision (3DV)*, 2026.

[17] Y. Chen, R. Hu, D. Xu, Z. Wang, L. Luo, and D. Li. Hidden Follower Detection via Refined Gaze and Walking State Estimation. In *Proc. IEEE Int. Conf. Multimedia and Expo (ICME)*, pp. 2081-2086, 2023."""

# Replace references section
idx = content.find('## References')
if idx > 0:
    content = content[:idx] + new_refs
    with open('report.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Done: 17 references, clean numbering')
else:
    print('ERROR: ## References not found')
