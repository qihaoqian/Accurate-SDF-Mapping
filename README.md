# Precise SDF Representation through Octree-Based Feature Priors and Hash-Encoded Networks

## Abstract
This report focuses on accurately predicting the SDF at any location using a continuous function, enhanced by octree-based prior feature interpolation and a hash-encoded neural network output.

## Installation

1. Begin by cloning this repository and all its submodules using the following command:

   ```bash
   git clone --recursive https://github.com/qihaoqian/Accurate-SDF-Mapping.git
   ```

2. Create an anaconda environment called `AccurateSDF`. 
   ```bash
   conda create -n AccurateSDF python=3.10
   ```

3. Install the [Pytorch](https://pytorch.org/) manually for your hardware platform.

4. Install the dependency packages.
   ```bash
   bash install.sh
   ```

5. Install tinycudann and its pytorch extension following https://github.com/NVlabs/tiny-cuda-nn 
   ```bash
   cd third_party/tinycudann
   cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
   cmake --build build --config RelWithDebInfo -j
   cd bindings/torch
   python setup.py install
   ```

## Run in dataset

1. Replace the filename in `src/mapping.py` with the built library

```bash
torch.classes.load_library("third_party/sparse_octree/build/lib.xxx/svo.xxx.so")
```

### Replica dataset

1. Download the data as below and the data is saved into the `./Datasets/Replica` folder.

```bash
bash scripts/download_replica.sh
```

```bash
# take replica room0 dataset as example
cd mapping
python -W ignore demo/run_mapping.py configs/replica/room_0.yaml
```

The final reconstructed mesh will be saved in `mapping/logs/{DATASET}/{DATA SEQUENCE}/{FILE_NAME}/mesh`.


## Evaluation

### Reconstruction Error

1. Download the ground truth Replica meshes 

```bash
bash scripts/download_replica_mesh.sh
```

2. Replace the filename in `eval/eval_recon.py` with the built library

```bash
torch.classes.load_library("third_party/sparse_octree/build/lib.xxx/svo.xxx.so")
```

3. Then run the command below. The 2D metric requires rendering of 1000 depth images. Use `-2d` to enable 2D metric. Use `-3d` to enable 3D metric. The reconstruction results will be saved in the `$OUTPUT_FOLDER`

```bash
# assign any output_folder and gt mesh you like, here is just an example
cd mapping
OUTPUT_FOLDER=logs/replica/room0/FILE_NAME
GT_MESH=../Datasets/Replica/cull_replica_mesh/room0.ply
python eval/eval_recon.py \
$OUTPUT_FOLDER/bak/config.yaml \
--rec_mesh $OUTPUT_FOLDER/mesh/final_mesh.ply \
--gt_mesh $GT_MESH \
--ckpt $OUTPUT_FOLDER/ckpt/final_ckpt.pth \
--out_dir $OUTPUT_FOLDER \
-2d \
-3d
```

