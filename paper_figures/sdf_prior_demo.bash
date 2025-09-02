#!/usr/bin/bash


SCRIPT=sdf_prior_demo.py

python3 $SCRIPT --load single_circle.yaml --no-title --no-interactive --output-png sdf-prior-single-circle.png
python3 $SCRIPT --load single_circle.yaml --draw-error --no-title --no-interactive --output-png sdf-prior-single-circle-error.png
python3 $SCRIPT --load single_circle.yaml --no-gradients --no-title --no-interactive --output-png sdf-prior-single-circle-no-gradients.png
python3 $SCRIPT --load single_circle.yaml --no-gradients --draw-error --no-title --no-interactive --output-png sdf-prior-single-circle-no-gradients-error.png
python3 $SCRIPT --load single_circle.yaml --draw-gt --no-title --no-interactive --output-png sdf-prior-single-circle-gt.png

python3 $SCRIPT --load four_circles.yaml --no-title --no-interactive --output-png sdf-prior-four-circles.png
python3 $SCRIPT --load four_circles.yaml --draw-error --no-title --no-interactive --output-png sdf-prior-four-circles-error.png
python3 $SCRIPT --load four_circles.yaml --no-gradients --no-title --no-interactive --output-png sdf-prior-four-circles-no-gradients.png
python3 $SCRIPT --load four_circles.yaml --no-gradients --draw-error --no-title --no-interactive --output-png sdf-prior-four-circles-no-gradients-error.png
python3 $SCRIPT --load four_circles.yaml --draw-gt --no-title --no-interactive --output-png sdf-prior-four-circles-gt.png
