# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

_ext_sources = glob.glob("src/*.cpp")

setup(
    name="sparse_octree",
    ext_modules=[
        CppExtension(
            name="sparse_octree.svo",
            sources=_ext_sources,
            extra_compile_args={"cxx": ["-g", "-O2", "-DNDEBUG"]},
        )
    ],
    packages=["sparse_octree"],
    cmdclass={"build_ext": BuildExtension},
)
