#!/bin/bash
# usage: run_ddp.sh NODE_RANK   (rank 0 hosts the dataset and is MASTER_ADDR)
# Required env: MASTER_ADDR (rank-0 IP on the fast link), NIC (interface name for NCCL/GLOO),
#               WORKDIR (contains hf/ models/ ultravox/ configs/ out/ logs/)
# Optional env: CONFIG (default ddp-whisper-small.yaml), IMAGE (default ultravox-train:tf581)
set -u
R=${1:?node rank}; : "${MASTER_ADDR:?}" "${NIC:?}" "${WORKDIR:?}"
CONFIG=${CONFIG:-ddp-whisper-small.yaml}; IMAGE=${IMAGE:-ultravox-train:tf581}
cd "$WORKDIR"; mkdir -p logs out; docker rm -f uvx-ddp >/dev/null 2>&1
docker run --name uvx-ddp --rm --network host --ipc=host --gpus all --device /dev/infiniband --cap-add IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 --memory=100g --memory-swap=100g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf -e PYTHONPATH=/uv -e PYTHONDONTWRITEBYTECODE=1 \
  -e ULTRAVOX_LLM_GRAD_CKPT=1 -e NCCL_SOCKET_IFNAME="$NIC" -e GLOO_SOCKET_IFNAME="$NIC" -e NCCL_DEBUG=WARN -e ULTRAVOX_LOCAL_HF_CACHE=/hf/hub -e HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} -e HF_DATASETS_OFFLINE=${HF_HUB_OFFLINE:-0} \
  -v "$PWD/hf:/hf" -v "$PWD/models:/models:ro" -v "$PWD/ultravox:/uv" -v "$PWD/configs:/cfg" -v "$PWD/out:/out" -w /uv "$IMAGE" \
  python3 -m torch.distributed.run --nnodes 2 --node_rank "$R" --nproc_per_node 1 \
  --master_addr "$MASTER_ADDR" --master_port "${MASTER_PORT:-29555}" \
  -m ultravox.training.train --config_path "/cfg/$CONFIG" --output_dir /out/run > logs/ddp-resume24000.log 2>&1
echo "EXIT $?" >> logs/ddp-resume24000.log
