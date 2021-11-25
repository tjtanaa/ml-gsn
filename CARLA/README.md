## Setup
This folder contains 4 repository for training of 4 synthetic datasets:
1.	CARLA_singleView is a model trained on self-generated single camera dataset from CARLA.
2.	CARLA_stereoView is a model trained on self-generated dual camera dataset from CARLA.
3.	KITTICARLA_singleView is a model trained on self-generated single camera KITTI-CARLA dataset.
4.	KITTICARLA_singleView is a model trained on self-generated DUAL camera KITTI-CARLA dataset.

The dataset can be downloaded from https://hkustconnect-my.sharepoint.com/:f:/g/personal/khleeba_connect_ust_hk/Epis1dORm-NCuZk6edqmgpwBcjIzDVJWGYWxV7itn_PVlw?e=zcvCV2

## Environment Setup
Install conda environment:
`conda env create -f environment.yaml python=3.6`

`conda activate gsn`

## Training:
1.	Modify the GPU device ID in `scripts/launch_gsn_vizdoom_64x64.sh`. 
2.	Modify the training setting in `configs/models/gsn_vizdoom_config.yaml` if want to try out other hyper-parameters.
3.	Put the corresponding preprocessed dataset downloaded from the drive to the `data/` folder.
4.	`bash scripts/launch_gsn_vizdoom_64x64.sh`

## Testing Pretrained Model (Qualitative Evaluation)
1.  Download the [_`log_dir`_](https://hkustconnect-my.sharepoint.com/:f:/g/personal/khleeba_connect_ust_hk/Epis1dORm-NCuZk6edqmgpwBcjIzDVJWGYWxV7itn_PVlw?e=zcvCV2) to the root of project (e.g. `CARLA/CARLA_singleView`). 
2.	Put the corresponding preprocessed dataset downloaded from the drive to the `data/` folder.
3.	Start Jupyter server. 
4.	Open `notebooks/walkthrough_demo.ipynb`.
5.	Change the absolute path inside to your corresponding absolute path
6.	Run it.

## Testing (Qualitative Evaluation):
1.	Put the corresponding preprocessed dataset downloaded from the drive to the `data/` folder.
2.	Start Jupyter server. 
3.	Open `notebooks/walkthrough_demo.ipynb`.
4.	Change the absolute path inside to your corresponding absolute path
5.	Run it.

## Model Training and Testing Information:
1.	Stored in logs.
2.	`logs/vis` give the visualization of the model training details.
