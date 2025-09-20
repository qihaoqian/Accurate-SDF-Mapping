#!/usr/bin/bash

SCRIPT_DIR=$(dirname "$0")

DATASET_DIR="/home/daizhirui/DataArchive/Replica-SDF-aug"

for scene in office0 office1 office2 office3 office4 room0 room1 room2; then

    PYTHONPATH="${SCRIPT_DIR}/.." python3 grad_sdf/dataset/generate_test_set.py \
        --mesh-path ${DATASET_DIR}/${scene}_mesh.ply \
        --grid-resolution 0.0125 \
        --eps 0.001 \
        --output-dir ${DATASET_DIR}/${scene}/test_set

done