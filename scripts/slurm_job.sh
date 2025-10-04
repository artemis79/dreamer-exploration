#!/bin/bash
#SBATCH --account=aip-mbowling
#SBATCH --cpus-per-task=1 
#SBATCH --gpus-per-node=1 
#SBATCH --mem=16G 
#SBATCH --time=0-2:59


echo "Starting task $SLURM_ARRAY_TASK_ID"
# SOCKS5 Proxy
if [ "$SLURM_TMPDIR" != "" ]; then
    echo "Setting up SOCKS5 proxy..."
    ssh -q -N -T -f -D 8888 `echo $SSH_CONNECTION | cut -d " " -f 3`
    export ALL_PROXY=socks5h://localhost:8888
fi
 
module load python/3.10 StdEnv/2023 gcc opencv/4.8.1 swig rust

cd $SLURM_TMPDIR

export ALL_PROXY=socks5h://localhost:8888
git config --global http.proxy 'socks5://127.0.0.1:8888'
git clone https://github.com/artemis79/dreamer-exploration.git

cd dreamer-exploration/

python -m venv .venv
source .venv/bin/activate

pip install requests[socks] --no-index
pip install .
pip install .[atari]


exp_name="dreamer_v3"
env="atari"
env_id="PongNoFrameskip-v4"
total_steps=1000

pip install "sheeprl[atari] @ git+https://github.com/Eclectic-Sheep/sheeprl.git"
python sheeprl.py exp=${exp_name} env=${env} env.id=${env_id} algo.cnn_keys.encoder="[rgb]" fabric.accelerator=cuda fabric.strategy=ddp fabric.devices=2 algo.mlp_keys.encoder="[]" algo.total_steps=${total_steps}


log_dir="$HOME/scratch/test_logs/${exp_name}/${env_id}"
if [ ! -d "$log_dir" ]; then
    mkdir $log_dir
fi

cp -r logs/runs/${exp_name}/${env_id}/2* ${log_dir}