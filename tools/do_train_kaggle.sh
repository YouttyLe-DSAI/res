set -x

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.0;7.5;8.0}"
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export DISABLE_TENSORBOARD=${DISABLE_TENSORBOARD:-1}
export PRETRAINED=${PRETRAINED:-1}
export KAGGLE_EPOCHS=${KAGGLE_EPOCHS:-80}
export KAGGLE_EVAL_PERIOD=${KAGGLE_EVAL_PERIOD:-5}
export USE_MSDA_PYTORCH=${USE_MSDA_PYTORCH:-1}

proj_path=$(pwd)
dir_name=$(dirname "$0")
export PYTHONPATH=$proj_path:$PYTHONPATH

cfg_path=${CFG_PATH:-projects/configs/citystreet/citystreet_voxel_kaggle_tiny.py}
seed=${SEED:-42}
apply_image_mask=${APPLY_IMAGE_MASK:-0}
deter=${DETER:-0}
amp=${AMP:-1}

resume=${RESUME:-0}
only_weight=${ONLY_WEIGHT:-0}
chkp_path=${CHKP_PATH:-}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python $dir_name/train.py \
    --seed=$seed \
    --apply_image_mask=$apply_image_mask \
    --deter=$deter \
    --cfg_path=$cfg_path \
    --resume=$resume \
    --chkp_path=$chkp_path \
    --only_weight=$only_weight \
    --amp=$amp
