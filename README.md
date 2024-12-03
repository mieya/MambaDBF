# MambaDBF
The repo is the official implementation for the paper: MambaDBF: Dual-Branch Mamba with FFN for Time Series Forecasting

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
   sh ./scripts/Weather/MambaDBF.sh
   ```
