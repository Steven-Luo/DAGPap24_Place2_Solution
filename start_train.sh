#!/bin/bash
export RANDOM_SEED=43

script=versions/$1

export NNODES=1
NPROC=`nvidia-smi | grep MiB | wc -l`

if [ -z "${NNODES}" ]; then
    export NNODES=4
fi

if [ ${NNODES} -eq 1 ]; then
    DISTRIBUTED_ARGS="
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC} \
    "
else
    DISTRIBUTED_ARGS="
        --nnodes=${NNODES} \
        --nproc_per_node=${NPROC} \
        --master_port=${MASTER_PORT} \
        --node_rank=${RANK} \
        --master_addr=${MASTER_ADDR} \
    "
fi

if [ ! -d logs ]; then
    mkdir logs
fi

echo "NNODES: ${NNODES}"
echo "NPROC: ${NPROC}"

torchrun ${DISTRIBUTED_ARGS} ${script}.py 2>&1 | tee logs/train_${script}_rank_${RANK}_$(date +"%Y%m%d_%H%M").log
