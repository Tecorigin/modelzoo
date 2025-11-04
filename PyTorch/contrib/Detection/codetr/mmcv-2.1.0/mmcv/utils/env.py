# Copyright (c) OpenMMLab. All rights reserved.
"""This file holding some environment constant for sharing by other files."""

import os.path as osp
import subprocess

import torch
from mmengine.utils.dl_utils import collect_env as mmengine_collect_env

import mmcv


def collect_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - SDAA available: Bool, indicating if SDAA is available.
            - GPU devices: Device type of each GPU.
            - SDAA_HOME (optional): The env var ``SDAA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - MSVC: Microsoft Virtual C++ Compiler version, Windows only.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.
            - MMEngine: MMEngine version.
            - MMCV: MMCV version.
            - MMCV Compiler: The GCC version for compiling MMCV ops.
            - MMCV SDAA Compiler: The SDAA version for compiling MMCV ops.
    """
    env_info = mmengine_collect_env()

    # MMEngine does not add the hipcc compiler information when collecting
    # environment information, so it is added here. When MMEngine v0.3.0 is
    # released, the code here can be removed.
    sdaa_available = torch.sdaa.is_available()
    if sdaa_available and env_info.get('NVCC') == 'Not Available':
        SDAA_HOME = env_info['SDAA_HOME']
        if SDAA_HOME is not None and osp.isdir(SDAA_HOME):
            if SDAA_HOME == '/opt/rocm':
                try:
                    nvcc = osp.join(SDAA_HOME, 'hip/bin/hipcc')
                    nvcc = subprocess.check_output(
                        f'"{nvcc}" --version', shell=True)
                    nvcc = nvcc.decode('utf-8').strip()
                    release = nvcc.rfind('HIP version:')
                    build = nvcc.rfind('')
                    nvcc = nvcc[release:build].strip()
                except subprocess.SubprocessError:
                    nvcc = 'Not Available'
            else:
                try:
                    nvcc = osp.join(SDAA_HOME, 'bin/nvcc')
                    nvcc = subprocess.check_output(f'"{nvcc}" -V', shell=True)
                    nvcc = nvcc.decode('utf-8').strip()
                    release = nvcc.rfind('Cuda compilation tools')
                    build = nvcc.rfind('Build ')
                    nvcc = nvcc[release:build].strip()
                except subprocess.SubprocessError:
                    nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

    env_info['MMCV'] = mmcv.__version__

    try:
        from mmcv.ops import get_compiler_version, get_compiling_sdaa_version
    except ModuleNotFoundError:
        env_info['MMCV Compiler'] = 'n/a'
        env_info['MMCV SDAA Compiler'] = 'n/a'
    else:
        env_info['MMCV Compiler'] = get_compiler_version()
        env_info['MMCV SDAA Compiler'] = get_compiling_sdaa_version()

    return env_info
