#ifndef PYTORCH_SDAA_HELPER
#define PYTORCH_SDAA_HELPER

// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>

// #include <ATen/cuda/CUDAApplyUtils.cuh>
// #include <THC/THCAtomics.cuh>

// #include "common_cuda_helper.hpp"
#include <torch/extension.h>
#include <torch_sdaa/sdaa_extension.h>
// #include "teco_hardswish.h"

using torch::Half;
using torch::Tensor;
using phalf = torch::Half;

#define __PHALF(x) (x)
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#endif  // PYTORCH_CUDA_HELPER
