#!/bin/bash
# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_TAG="example-algorithm-segrap2025-task1"
DOCKER_NOOP_VOLUME="${DOCKER_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"
MEM_LIMIT="30g"        # Reduced memory
SHM_SIZE="2g"           # Shared memory for nnUNet
USE_GPUS="--gpus all"  # Enable GPU support

echo "+++++ Cleaning up the output folder +++++"
if [ -d "$OUTPUT_DIR" ]; then
  rm -rf "${OUTPUT_DIR}"/*
  chmod -f o+rwx "$OUTPUT_DIR"
else
  mkdir -p "$OUTPUT_DIR"
  chmod o+rwx "$OUTPUT_DIR"
fi

echo "+++++ (Re)building the container +++++"
docker build "$SCRIPT_DIR" \
  --platform=linux/amd64 \
  --tag $DOCKER_TAG

echo "+++++ Doing a forward pass +++++"
docker volume create "$DOCKER_NOOP_VOLUME"

docker run --rm \
    --memory="${MEM_LIMIT}" \
    --memory-swap="${MEM_LIMIT}" \
    --shm-size="${SHM_SIZE}" \
    --platform=linux/amd64 \
    --network="none" \
    --cap-drop="ALL" \
    --security-opt="no-new-privileges" \
    $USE_GPUS \
    --volume "$INPUT_DIR":/input \
    --volume "$OUTPUT_DIR":/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    $DOCKER_TAG

docker volume rm "$DOCKER_NOOP_VOLUME"

# Fix permissions on host
HOST_UID=$(id -u)
HOST_GID=$(id -g)
sudo chown -R $HOST_UID:$HOST_GID "$OUTPUT_DIR"

echo "+++++ Wrote results to ${OUTPUT_DIR} +++++"
