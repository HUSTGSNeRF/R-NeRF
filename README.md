# R-NeRF: Neural Radiance Fields for Modeling RIS-enabled Wireless Environments
[Paper](https://arxiv.org/abs/2405.11541)<br>
Thank you for your interest in our work. This repository mainly maintains the code of R-NeRF. We employ NeRF-based ray tracing techniques to capture the dynamic complexity of signal propagation, and propose an R-NeRF model for a wireless environment supported by intelligent metasurface RIS. Specifically, we introduced a well-designed two-stage framework to accurately trace the entire transmission path. Each stage in our framework accurately characterizes the dynamics of electromagnetic signal propagation.<br>
If you find the code useful, please refer to our work using:<br>
@misc{yang2024rnerfneuralradiancefields,
      title={R-NeRF: Neural Radiance Fields for Modeling RIS-enabled Wireless Environments}, 
      author={Huiying Yang and Zihan Jin and Chenhao Wu and Rujing Xiong and Robert Caiming Qiu and Zenan Ling},
      year={2024},
      eprint={2405.11541},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2405.11541}, 
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
