"""Fix references in both reports."""
import sys

new_refs_en = '''
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

[11] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra. Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. In *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, pp. 618-626, 2017.

[12] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research*, 15(1):1929-1958, 2014.

[13] I. Loshchilov and F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. In *Proc. Int. Conf. Learning Representations (ICLR)*, 2017.

[14] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet Classification with Deep Convolutional Neural Networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 1097-1105, 2012.

[15] K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image Recognition. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 770-778, 2016.

[16] A. Paszke, S. Gross, F. Massa, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In *Advances in Neural Information Processing Systems (NeurIPS)*, pp. 8024-8035, 2019.

[17] W. Falcon et al. PyTorch Lightning. 2019. https://github.com/Lightning-AI/lightning.

[18] D. P. Kingma and J. Ba. Adam: A Method for Stochastic Optimization. In *Proc. Int. Conf. Learning Representations (ICLR)*, 2015.

[19] T. Baltrusaitis, P. Robinson, and L.-P. Morency. OpenFace: An Open Source Facial Behavior Analysis Toolkit. In *Proc. IEEE Winter Conf. Applications of Computer Vision (WACV)*, pp. 1-10, 2016.

[20] M. Hayhoe and D. Ballard. Eye Movements in Natural Behavior. *Trends in Cognitive Sciences*, 9(4):188-194, 2005.

[21] K. A. Funes Mora and J.-M. Odobez. Gaze Estimation from Multimodal Kinect Data. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 25-30, 2012.

[22] A. Recasens, C. Vondrick, A. Khosla, and A. Torralba. Following Gaze in Video. In *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, pp. 1435-1443, 2017.

[23] E. Chong, N. Ruiz, Y. Wang, Y. Zhang, A. Rozga, and J. M. Rehg. Connecting Gaze, Scene, and Attention: Generalized Attention Estimation via Joint Modeling of Gaze and Scene Saliency. In *Proc. European Conf. Computer Vision (ECCV)*, pp. 383-398, 2018.

[24] A. Doshi and M. M. Trivedi. Head and Gaze Dynamics in Visual Attention and Context Learning. In *Proc. IEEE CVPR Workshops*, pp. 77-84, 2009.

[25] A. Fathi, J. K. Hodgins, and J. M. Rehg. Social Interactions: A First-Person Perspective. In *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 1226-1233, 2012.

[26] E. Lan, Z. Hu, and J. Liu. UAGE: A Supervised Contrastive Method for Unconstrained Adaptive Gaze Estimation. In *Proc. Asian Conf. Computer Vision (ACCV)*, 2024.

[27] GazeD: Context-Aware Diffusion for Accurate 3D Gaze Estimation. *arXiv preprint*, 2023.
'''

for fname in ['report.md', 'report_cn.md']:
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            content = f.read()
        idx = content.find('\n## References')
        if idx < 0:
            idx = content.find('\n## 参考文献')
        if idx > 0:
            content = content[:idx+1] + '## References' + new_refs_en
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'{fname}: OK ({len(new_refs_en.split(chr(10)))} lines of refs)')
        else:
            print(f'{fname}: References section not found')
    except Exception as e:
        print(f'{fname}: {e}')
