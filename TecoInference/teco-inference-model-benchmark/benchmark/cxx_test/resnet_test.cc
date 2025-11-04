// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.
#include <teco_infer/common/common.h>
#include <teco_infer/core/device_type.h>
#include <teco_infer/core/teco_interface_api.h>
#include <teco_infer/utils/tensor/empty.h>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#ifdef _DEBUG
print("debug!");
#else

#endif

void resnet_test(const std::string& engine_path, const int& depth = 2, const int& run_loops = 10) {
  TECO_INFER_NS::Engine engine(engine_path);
  auto ctx = engine.CreateContext();
  // 3核组
  TECO_INFER_NS::TensorShape target_shape({192, 3, 224, 224});
  std::optional<TECO_INFER_NS::Device> dev("cpu");
  auto tensor_options =
      TECO_INFER_NS::TensorOptions().device(dev).dtype(TECO_INFER_NS::kFloat).pinned_memory(false);
  std::vector<TECO_INFER_NS::Tensor> tensors(1);
  std::vector<TECO_INFER_NS::Future> fu_queue(run_loops);
  auto begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < run_loops; ++i) {
    tensors[0] = empty(target_shape, tensor_options);
    fu_queue[i] = ctx->RunAsync(tensors);
    if (i >= depth - 1) {
      fu_queue[i].get();
    }
  }
  for (int i = 0; i < depth; ++i) {
    fu_queue[run_loops - depth + i].get();
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_span = std::chrono::duration<double, std::milli>(end - begin).count();
  tensors.clear();
  fu_queue.clear();
  std::cout << "Run " << run_loops << " inferences request on " << time_span
            << " ms, average inference time is " << time_span / run_loops << " ms." << std::endl;
  ctx->Release();
  engine.Release();
}

int main(int argc, char* argv[]) {
  std::string engine_path = "/mnt/wzl/resnet_64.tecoengine";
  int depth = 2;
  int run_loops = 50;
  resnet_test(engine_path, depth, run_loops);

  return 0;
}
