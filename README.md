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
   pip install -r requirements.txt
   ```

5. Install tinycudann and its pytorch extension following https://github.com/NVlabs/tiny-cuda-nn
   ```bash
   cd third_party/tinycudann
   cmake . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release -j`nproc`
   cd bindings/torch
   python setup.py install
   cd ../../../..
   cd third_party/sparse_octree
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
python demo/run_mapping.py configs/replica/room_0.yaml --save-mesh
```

The final reconstructed mesh will be saved in `mapping/logs/{DATASET}/{DATA SEQUENCE}/{FILE_NAME}/mesh`.


## Evaluation

### Reconstruction Error

1. Download the ground truth Replica meshes

```bash
bash scripts/download_replica_mesh.sh
```

```bash
# assign any output_folder and gt mesh you like, here is just an example
OUTPUT_FOLDER=logs/replica/room0/FILE_NAME
GT_MESH=../Datasets/Replica/cull_replica_mesh/room0.ply
python src/evaluate.py logs/replica/room0/h2mapping-baseline/bak/config.yaml # --save mesh
```
