# R-NeRF: Neural Radiance Fields for Modeling RIS-enabled Wireless Environments
[Paper](https://arxiv.org/abs/2405.11541)<br>

Thank you for your interest in our work. This repository mainly maintains the code of R-NeRF. We employ NeRF-based ray tracing techniques to capture the dynamic complexity of signal propagation, and propose an R-NeRF model for a wireless environment supported by intelligent metasurface RIS. Specifically, we introduced a well-designed two-stage framework to accurately trace the entire transmission path. Each stage in our framework accurately characterizes the dynamics of electromagnetic signal propagation.<br>
<br>
If you find the code useful, please refer to our work using:<br>
```bibtex
@misc{yang2024rnerfneuralradiancefields,
      title={R-NeRF: Neural Radiance Fields for Modeling RIS-enabled Wireless Environments}, 
      author={Huiying Yang and Zihan Jin and Chenhao Wu and Rujing Xiong and Robert Caiming Qiu and Zenan Ling},
      year={2024},
      eprint={2405.11541},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2405.11541}, 
}
```
## Presentation of the document<br>
**Project Structure** <br>
```python
---Measured data experiments:    #  This file contains the measured data set, which is mainly used for experiments
  ---R_NeRF.py                   # run R_NeRF.py to get the result
  ---model.py                    # contains R_NeRF model****
  ---dataloader.py.              # Contains various processing methods for data
  ---renderer.py                 # Contains R-Nerf's rendering method for data
  ---conda_env.yml               # Contains relevant configuration information for this code
  ---configs:    
    ---config.yml                # Contains parameter settings related to code runtime
  ---data/BLE:    
    ---gateway_positionris.yml   # Contains RIS coordinate information
    ---gateway_positionrx.yml    # Contains receiver information
    ---gateway_positiontx.yml    # Contains transmitterinformation
    ---ris_pos.csv               # Contains RIS coordinate information
    ---rx_pos.csv                # Contains receiver information
    ---tx_pos.csv                # Contains transmitterinformation
    ---train_index.txt           # Contains training dataset index information
    ---test_index.txt            # Contains testing dataset index information
  ---data_index:                 # Contains data index information at different sampling rates
  ---logs/BLE/###：              # Contains corresponding experimental result information
  ---utils：                     # Used to record and manage log information of programs
    ---logger.py

---Simulation data experiments:  # The file content is similar to the appeal content
```
**Measured data experiments** <br>
This file contains the measured data set, which is mainly used for experiments<br>
<br>
**Simulation data experiment** <br>
This file contains the simulation data set, and the simulation data is mainly aexperimented<br>
## Configuration Information <br>
The relevant package and configuration information can be found in **conda_env.yml**.
## training the model<br>
python R_NeRF.py --mode train --config configs/config.yml --dataset_type ble --gpu 0 <br>
## Inference the model<br>
python R_NeRF.py --mode test --config configs/config.yml --dataset_type ble --gpu 0<br>
