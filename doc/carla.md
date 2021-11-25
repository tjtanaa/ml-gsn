# Instruction of Generating Sequences using Carla

1. Download Carla release:
    https://github.com/carla-simulator/carla/releases
    On Ubuntu: CARLA_0.9.12.tar.gz
2. Decompress the downloaded files
3. Install python dependency, run:
    cd PythonAPI/examples
    python3 -m pip install -r requirements.txt
4. Download the script of generating GSN-style dataset:
    https://hkustconnect-my.sharepoint.com/:f:/g/personal/kcshum_connect_ust_hk/Eo5n_U-QKvxLskjexcRlxxQBDzdFyNRMQMO_BXnI3trHlg?e=QiZGDV
    the script is 'GSN.py' and place it to PythonAPI/examples
5. Go to root directory, run:
    ./CarlaUE4.sh
6. Open a new terminal, go to 'PythonAPI/examples', run:
    python3 GSN.py -s 4 --res 64x64

    The output will be generated in folder 'PythonAPI/examples/_out'
    note: you can check main() function for available input parameters.
          Here -s is the seed. --res is resolution.

# Training and Testing
Please refer to [`CARLA/README.md`](../CARLA/README.md).