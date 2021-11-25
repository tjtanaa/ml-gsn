## Setup
This folder contains 4 repository for training of 4 synthetic datasets:
1.	CARLA_singleView is a model trained on self-generated single camera dataset from CARLA.
2.	CARLA_stereoView is a model trained on self-generated dual camera dataset from CARLA.
3.	KITTICARLA_singleView is a model trained on self-generated single camera KITTI-CARLA dataset.
4.	KITTICARLA_singleView is a model trained on self-generated DUAL camera KITTI-CARLA dataset.

The dataset can be downloaded from https://hkustconnect-my.sharepoint.com/:u:/g/personal/khleeba_connect_ust_hk/EVHZ_qshvBdChx4h1TFaC0gBwv1Ab6antkk-jBRMUNpAyg?e=15t1X7 

## Environment Setup
Install conda environment:
`conda env create -f environment.yaml python=3.6`

`conda activate gsn`

## Training:
1.	Modify the GPU device ID in `scripts/launch_gsn_vizdoom_64x64.sh`. 
2.	Modify the training setting in `configs/models/gsn_vizdoom_config.yaml` if want to try out other hyper-parameters.
3.	Put the corresponding preprocessed dataset downloaded from the drive to the `data/` folder.
4.	`bash scripts/launch_gsn_vizdoom_64x64.sh`


## Testing:
1.	Put the corresponding preprocessed dataset downloaded from the drive to the `data/` folder.
2.	Start Jupyter server. 
3.	Open `notebooks/walkthrough_demo.ipynb`.
4.	Change the absolute path inside to your corresponding absolute path
5.	Run it.

## Model Training and Testing Information:
1.	Stored in logs.
2.	`logs/vis` give the visualization of the model training details.
