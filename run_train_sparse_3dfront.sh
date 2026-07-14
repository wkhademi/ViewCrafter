#!/bin/bash

name="train_sparse_3dfront_1024"
config_file=configs/train_sparse_3dfront_1024.yaml
save_root="./logs"

# Override before calling this script for a real multi-GPU/multi-node run, e.g.:
#   HOST_GPU_NUM=8 HOST_NUM=1 bash run_train_sparse_3dfront.sh
HOST_GPU_NUM=${HOST_GPU_NUM:-1}
HOST_NUM=${HOST_NUM:-1}

# Default to the first HOST_GPU_NUM devices if not explicitly set - without
# this, CUDA_VISIBLE_DEVICES would default to just GPU 0 regardless of
# HOST_GPU_NUM, and devices 1..N-1 would be invisible to the launched
# processes. Override CUDA_VISIBLE_DEVICES yourself if you want specific
# (e.g. non-contiguous) GPUs instead.
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((HOST_GPU_NUM - 1)))
fi

mkdir -p $save_root/$name

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -m torch.distributed.launch \
--nproc_per_node=$HOST_GPU_NUM --nnodes=$HOST_NUM --master_addr=127.0.0.1 --master_port=12352 --node_rank=0 \
main/trainer.py \
--base $config_file \
--train \
--name $name \
--logdir $save_root \
--devices $HOST_GPU_NUM \
lightning.trainer.num_nodes=$HOST_NUM
