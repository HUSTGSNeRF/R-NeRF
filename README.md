# R-NeRF: Neural Radiance Fields for Modeling RIS-enabled Wireless Environments
[Paper](https://arxiv.org/abs/2405.11541)<br>
Thank you for your interest in our work. This repository mainly maintains the code of R-NeRF. We employ NeRF-based ray tracing techniques to capture the dynamic complexity of signal propagation, and propose an R-NeRF model for a wireless environment supported by intelligent metasurface RIS. Specifically, we introduced a well-designed two-stage framework to accurately trace the entire transmission path. Each stage in our framework accurately characterizes the dynamics of electromagnetic signal propagation.<br>
If you find the code useful, please refer to our work using:<br>
```bibtex
@InProceedings{pmlr-v235-ling24a,
  title = {Deep Equilibrium Models are Almost Equivalent to Not-so-deep Explicit Models for High},
  author = {Ling, Zenan and Li, Longbo and Feng, Zhanbo and Zhang, Yixuan and Zhou, Feng and Qiu, ...},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages = {30585--30609},
  year = {2024},
  editor = {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and O...},
  volume = {235},
  series = {Proceedings of Machine Learning Research},
  month = {21--27 Jul},
  publisher = {PMLR},
  pdf = {https://raw.githubusercontent.com/mlresearch/v235/main/assets/ling24a/ling24a.pdf},
  url = {https://proceedings.mlr.press/v235/ling24a.html},
}
## Presentation of the document<br>
**Measured data experiments** <br>
This file contains the measured data set, which is mainly used for experiments<br>
**Simulation data experiment** <br>
This file contains the simulation data set, and the simulation data is mainly experimented<br>
## Configuration Information <br>
The relevant package and configuration information can be found in **conda_env.yml**.
## training the model<br>
python R_NeRF.py --mode train --config configs/config.yml --dataset_type ble --gpu 0 <br>
## Inference the model<br>
python R_NeRF.py --mode test --config configs/config.yml --dataset_type ble --gpu 0<br>
