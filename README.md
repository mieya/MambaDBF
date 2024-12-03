# MambaDBF
The repo is the official implementation for the paper: MambaDBF: Dual-Branch Mamba with FFN for Time Series Forecasting

Key codes:

* For the architecture design of MambaDBF, please refer primarily to `models/MambaDBF.py`.
* For Weighted Signal Decay Loss (EWSDL), please focus on the `exp/Exp_Long_Term_Forecast_EWSDL.py`.
* For MambaFFN, please refer mainly to `layers/Mambaffn.py`.
* 

## Usage
1. Install Python 3.8. For convenience, execute the following command.

   ```shell
   pip install -r requirements.txt 
   ```
2. For setting up the Mamba environment, please refer to https://github.com/state-spaces/mamba. Here is a simple instruction on Linux system,

   ```
   pip install causal-conv1d>=1.2.0
   pip install mamba-ssm
   ```
3. Train and evaluate model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

   ```bash
   sh ./scripts/MambaDBF_scripts/MambaDBF_Weather.sh
   ```
