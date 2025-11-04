# Copyright (c) OpenMMLab. All rights reserved.
"""This file holding some environment constant for sharing by other files."""
import os
import os.path as osp
import subprocess
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch_sdaa

import mmengine
from mmengine.device import is_sdaa_available, is_musa_available
from .parrots_wrapper import TORCH_VERSION, get_build_config, is_rocm_pytorch


def _get_sdaa_home():
    if TORCH_VERSION == 'parrots':
        from parrots.utils.build_extension import SDAA_HOME
    else:
        if is_rocm_pytorch():
            from torch_sdaa.utils.cpp_extension import ROCM_HOME
            SDAA_HOME = ROCM_HOME
        else:
            from torch_sdaa.utils.cpp_extension import SDAA_HOME
    return SDAA_HOME


def _get_musa_home():
    return os.environ.get('MUSA_HOME')


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
            - OpenCV (optional): OpenCV version.
            - MMENGINE: MMENGINE version.
    """
    from distutils import errors

    env_info = OrderedDict()
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    sdaa_available = is_sdaa_available()
    musa_available = is_musa_available()
    env_info['SDAA available'] = sdaa_available
    env_info['MUSA available'] = musa_available
    env_info['numpy_random_seed'] = np.random.get_state()[1][0]

    if sdaa_available:
        devices = defaultdict(list)
        for k in range(torch.sdaa.device_count()):
            # devices[torch.sdaa.get_device_name(k)].append(str(k))
            devices['TECO_AICARD_01'].append(str(k))           
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

        SDAA_HOME = _get_sdaa_home()
        env_info['SDAA_HOME'] = SDAA_HOME

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
                    release = nvcc.rfind('sdaa compilation tools')
                    build = nvcc.rfind('Build ')
                    nvcc = nvcc[release:build].strip()
                except subprocess.SubprocessError:
                    nvcc = 'Not Available'
            env_info['NVCC'] = nvcc
    elif musa_available:
        devices = defaultdict(list)
        for k in range(torch.musa.device_count()):
            devices[torch.musa.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

        MUSA_HOME = _get_musa_home()
        env_info['MUSA_HOME'] = MUSA_HOME

        if MUSA_HOME is not None and osp.isdir(MUSA_HOME):
            try:
                mcc = osp.join(MUSA_HOME, 'bin/mcc')
                subprocess.check_output(f'"{mcc}" -v', shell=True)
            except subprocess.SubprocessError:
                mcc = 'Not Available'
            env_info['mcc'] = mcc
    try:
        # Check C++ Compiler.
        # For Unix-like, sysconfig has 'CC' variable like 'gcc -pthread ...',
        # indicating the compiler used, we use this to get the compiler name
        import io
        import sysconfig
        cc = sysconfig.get_config_var('CC')
        if cc:
            cc = osp.basename(cc.split()[0])
            cc_info = subprocess.check_output(f'{cc} --version', shell=True)
            env_info['GCC'] = cc_info.decode('utf-8').partition(
                '\n')[0].strip()
        else:
            # on Windows, cl.exe is not in PATH. We need to find the path.
            # distutils.ccompiler.new_compiler() returns a msvccompiler
            # object and after initialization, path to cl.exe is found.
            import locale
            import os
            from distutils.ccompiler import new_compiler
            ccompiler = new_compiler()
            ccompiler.initialize()
            cc = subprocess.check_output(
                f'{ccompiler.cc}', stderr=subprocess.STDOUT, shell=True)
            encoding = os.device_encoding(
                sys.stdout.fileno()) or locale.getpreferredencoding()
            env_info['MSVC'] = cc.decode(encoding).partition('\n')[0].strip()
            env_info['GCC'] = 'n/a'
    except (subprocess.CalledProcessError, errors.DistutilsPlatformError):
        env_info['GCC'] = 'n/a'
    except io.UnsupportedOperation as e:
        # JupyterLab on Windows changes sys.stdout, which has no `fileno` attr
        # Refer to: https://github.com/open-mmlab/mmengine/issues/931
        # TODO: find a solution to get compiler info in Windows JupyterLab,
        # while preserving backward-compatibility in other systems.
        env_info['MSVC'] = f'n/a, reason: {str(e)}'

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = get_build_config()

    try:
        import torchvision
        env_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    try:
        import cv2
        env_info['OpenCV'] = cv2.__version__
    except ImportError:
        pass

    env_info['MMEngine'] = mmengine.__version__

    return env_info
