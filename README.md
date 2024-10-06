# R-NeRF
Thank you for your interest in our work. This repository mainly maintains the code of R-NeRF. We employ NeRF-based ray tracing techniques to capture the dynamic complexity of signal propagation, and propose an R-NeRF model for a wireless environment supported by intelligent metasurface RIS. Specifically, we introduced a well-designed two-stage framework to accurately trace the entire transmission path. Each stage in our framework accurately characterizes the dynamics of electromagnetic signal propagation.<br>
## Presentation of the document<br>
** Measured data experiments<br>
This file contains the measured data set, which is mainly used for experiments<br>
** Simulation data experiments<br>
This file contains the simulation data set, and the simulation data is mainly experimented<br>
** nerfxiaorong<br>
Contains code for partial ablation experiments
## training the model<br>
python nerf2_runner.py --mode train --config configs/ble-rssi.yml --dataset_type ble --gpu 0 <br>
## Inference the model<br>
python nerf2_runner.py --mode test --config configs/ble-rssi.yml --dataset_type ble --gpu 0<br>