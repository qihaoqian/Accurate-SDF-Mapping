#!/usr/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
python ${SCRIPT_DIR}/../ego_sdf/dataset/replica_augment_views.py \
    --original-dir /home/daizhirui/DataArchive/Replica-SDF \
    --output-dir /home/daizhirui/DataArchive/Replica-SDF-aug \
    --interval 50
