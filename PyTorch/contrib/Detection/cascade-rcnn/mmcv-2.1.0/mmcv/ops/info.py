# Copyright (c) OpenMMLab. All rights reserved.
import torch

if torch.__version__ == 'parrots':
    import parrots

    def get_compiler_version():
        return 'GCC ' + parrots.version.compiler

    def get_compiling_sdaa_version():
        return parrots.version.sdaa
else:
    from ..utils import ext_loader
    # ext_module = ext_loader.load_ext(
    #     '_ext', ['get_compiler_version', 'get_compiling_sdaa_version'])
    ext_module = ext_loader.load_ext(
        '_ext', ['get_compiler_version'])

    def get_compiler_version():
        return ext_module.get_compiler_version()

    def get_compiling_sdaa_version():
        return ext_module.get_compiling_sdaa_version()
