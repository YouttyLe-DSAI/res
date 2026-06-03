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
sample_idx=${SAMPLE_IDX:-0}
topk=${TOPK:-1200}
out=${OUT_HTML:-workdirs/kaggle_3d_visualization.html}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python $dir_name/visualize_kaggle_3d.py \
    --cfg_path=$cfg_path \
    --chkp_path="${CHKP_PATH:-}" \
    --part=${PART:-val} \
    --sample_idx=$sample_idx \
    --topk=$topk \
    --out="$out"
