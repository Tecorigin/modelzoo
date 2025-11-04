// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_sdaa_helper.hpp"
#include "roi_align_sdaa_kernel.h"

void ROIAlignForwardSDAAKernelLauncher(Tensor input, Tensor rois, Tensor output,
                                       Tensor argmax_y, Tensor argmax_x,
                                       int aligned_height, int aligned_width,
                                       float spatial_scale, int sampling_ratio,
                                       int pool_mode, bool aligned) {
  int output_size = output.numel();
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  at::sdaa::SDAAGuard device_guard(input.device());
  // sdaaStream_t stream = at::sdaa::getCurrentSDAAStream();
  sdaaStream_t stream;
  sdaaStreamCreate(&stream);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "roi_align_forward_sdaa_kernel", [&] {
        roi_align_forward_sdaa_kernel<scalar_t><<<blocks, threads>>>(
                output_size, input.data_ptr<scalar_t>(),
                rois.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                argmax_y.data_ptr<scalar_t>(), argmax_x.data_ptr<scalar_t>(),
                aligned_height, aligned_width,
                static_cast<scalar_t>(spatial_scale), sampling_ratio, pool_mode,
                aligned, channels, height, width);
      });

  // AT_SDAA_CHECK(sdaaGetLastError());
}

void ROIAlignBackwardSDAAKernelLauncher(Tensor grad_output, Tensor rois,
                                        Tensor argmax_y, Tensor argmax_x,
                                        Tensor grad_input, int aligned_height,
                                        int aligned_width, float spatial_scale,
                                        int sampling_ratio, int pool_mode,
                                        bool aligned) {
  int output_size = grad_output.numel();
  int channels = grad_input.size(1);
  int height = grad_input.size(2);
  int width = grad_input.size(3);

  // at::sdaa::SDAAGuard device_guard(grad_output.device());
  // sdaaStream_t stream = at::sdaa::getCurrentSDAAStream();
  sdaaStream_t stream;
  sdaaStreamCreate(&stream);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "roi_align_backward_sdaa_kernel", [&] {
        roi_align_backward_sdaa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                output_size, grad_output.data_ptr<scalar_t>(),
                rois.data_ptr<scalar_t>(), argmax_y.data_ptr<scalar_t>(),
                argmax_x.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
                aligned_height, aligned_width,
                static_cast<scalar_t>(spatial_scale), sampling_ratio, pool_mode,
                aligned, channels, height, width);
      });

  // AT_SDAA_CHECK(sdaaGetLastError());
}
