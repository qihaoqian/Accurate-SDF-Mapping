#!/usr/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPT=${SCRIPT_DIR}/mesh_visualization.py
MESH_DIR=/home/daizhirui/D/GoogleDrive/Documents/UCSD/Research/ERL/SDF/Neural-SDF/reconstructed_mesh_result

#for s in room0 room1 room2 office0 office1 office2 office3 office4; do
#for s in room0; do
#    OUTPUT_DIR=${SCRIPT_DIR}/mesh_vis/${s}
#    mkdir -p ${OUTPUT_DIR}
#
#    for f in gt h2mapping pin-slam hio voxblox; do
#        python3 ${SCRIPT} \
#            --cam-config-fp ${SCRIPT_DIR}/${s}_vedo_cam_config.py \
#            --lighting-config-fp ${SCRIPT_DIR}/${s}_vedo_lighting_config.py \
#            --mesh-fp ${MESH_DIR}/${s}-${f}.ply \
#            --output-img-fp ${OUTPUT_DIR}/${f}.png
#    done
#
#    for f in our our-prior; do
#        python3 ${SCRIPT} \
#            --cam-config-fp ${SCRIPT_DIR}/${s}_vedo_cam_config.py \
#            --lighting-config-fp ${SCRIPT_DIR}/${s}_vedo_lighting_config.py \
#            --mesh-fp ${MESH_DIR}/${s}-${f}.obj \
#            --output-img-fp ${OUTPUT_DIR}/${f}.png
#    done
#
#done

python3 ${SCRIPT} \
    --cam-config-fp ${SCRIPT_DIR}/room2_vedo_cam_config.py \
    --lighting-config-fp ${SCRIPT_DIR}/room2_vedo_lighting_config.py \
    --mesh-fp ${MESH_DIR}/room2-gt.ply \
    --output-img-fp ${SCRIPT_DIR}/mesh_vis/room2/gt.png \
    --image-size 1440 1440

#python3 ${SCRIPT} \
#    --cam-config-fp ${SCRIPT_DIR}/room2_vedo_cam_config.py \
#    --lighting-config-fp ${SCRIPT_DIR}/room2_vedo_lighting_config.py \
#    --mesh-fp ${MESH_DIR}/room2-our-prior.obj \
#    --output-img-fp ${SCRIPT_DIR}/mesh_vis/room2/our-prior.png \
#    --image-size 1440 1440
#
#python3 ${SCRIPT} \
#    --cam-config-fp ${SCRIPT_DIR}/room2_vedo_cam_config.py \
#    --lighting-config-fp ${SCRIPT_DIR}/room2_vedo_lighting_config.py \
#    --mesh-fp ${MESH_DIR}/room2-our.obj \
#    --output-img-fp ${SCRIPT_DIR}/mesh_vis/room2/our.png \
#    --image-size 1440 1440
