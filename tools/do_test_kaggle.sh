set -x

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export PRETRAINED=${PRETRAINED:-0}
export USE_MSDA_PYTORCH=${USE_MSDA_PYTORCH:-1}

proj_path=$(pwd)
dir_name=$(dirname "$0")
export PYTHONPATH=$proj_path:$PYTHONPATH

cfg_path=${CFG_PATH:-projects/configs/citystreet/citystreet_voxel_kaggle_tiny.py}
part=${PART:-val}
run_dir=${RUN_DIR:-$(ls -td workdirs/citystreet/vox/kaggle_scale06_layers1223/* 2>/dev/null | head -1)}
chkp_path=${CHKP_PATH:-$run_dir/checkpoints/best_mae_bev.pth}

if [ ! -f "$chkp_path" ]; then
    chkp_path=$(ls -t "$run_dir"/checkpoints/*.pth | head -1)
fi

echo "Evaluating checkpoint: $chkp_path"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python $dir_name/test.py \
    --cfg_path=$cfg_path \
    --part=$part \
    --chkp_path="$chkp_path"
