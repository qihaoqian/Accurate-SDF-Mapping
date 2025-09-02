#!/usr/bin/bash

SCRIPT=semi_sparse_structure_demo.py

python3 $SCRIPT --output-png interpolation_of_sparse_octree.png
python3 $SCRIPT --show-error --output-png interpolation_error_of_sparse_octree.png

python3 $SCRIPT --semi-sparse --output-png interpolation_of_semi_sparse_octree.png
python3 $SCRIPT --semi-sparse --show-error --output-png interpolation_error_of_semi_sparse_octree.png

python3 $SCRIPT --with-grad --output-png gradient_augmented_interpolation_of_sparse_octree.png
python3 $SCRIPT --with-grad --show-error --output-png gradient_augmented_interpolation_error_of_sparse_octree.png

python3 $SCRIPT --semi-sparse --with-grad --output-png gradient_augmented_interpolation_of_semi_sparse_octree.png
python3 $SCRIPT --semi-sparse --with-grad --show-error --output-png gradient_augmented_interpolation_error_of_semi_sparse_octree.png
