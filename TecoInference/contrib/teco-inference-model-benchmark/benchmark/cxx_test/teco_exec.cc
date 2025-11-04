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

#include "teco_exec.h"
#include <teco_infer/common/common.h>
#include <teco_infer/core/device_type.h>
#include <teco_infer/core/teco_interface_api.h>
#include <teco_infer/utils/tensor/empty.h>

#include <cctype>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <queue>

namespace TECO_INFER_NS {
std::unordered_map<std::string, TECO_INFER_NS::ScalarType> teco_exec_dtype_map{
    {"bool", TECO_INFER_NS::kBool},
    {"fp16", TECO_INFER_NS::kHalf},
    {"fp32", TECO_INFER_NS::kFloat},
    {"int32", TECO_INFER_NS::kInt},
    {"int64", TECO_INFER_NS::kLong}};

void TecoExec::TecoRun(const TecoExecArgConfig& config) {
  TECO_INFER_NS::Engine engine(config.engine_path);
  auto ctx = engine.CreateContext();
  auto tensors = GetInputTensor(
      const_cast<std::vector<std::vector<int64_t>>*>(&config.shapes),
      config.dtypes);
  // warm up
  for (int i = 0; i < config.warmup; ++i) {
    auto fu = ctx->RunAsync(tensors);
    fu.get();
  }
  // latency
  double latency_time = 0.0;
  for (int i = 0; i < config.iterations; ++i) {
    auto begin = std::chrono::high_resolution_clock::now();
    auto fu = ctx->RunAsync(tensors);
    fu.get();
    auto end = std::chrono::high_resolution_clock::now();
  latency_time +=
      std::chrono::duration<double, std::milli>(end - begin).count();
  }
  std::queue<TECO_INFER_NS::Future> fu_queue;
  auto begin = std::chrono::high_resolution_clock::now();
  if (config.run_async) {
    for (int i = 0; i < config.iterations; ++i) {
      fu_queue.push(ctx->RunAsync(tensors));
      if (i >= config.depth - 1) {
        auto future = fu_queue.front();
        future.get();
        fu_queue.pop();
      }
    }
    for (int i = 1; i < config.depth; ++i) {
      auto future = fu_queue.front();
      future.get();
      fu_queue.pop();
    }
  } else {  // run sync
    for (int i = 0; i < config.iterations; ++i) {
      auto fu = ctx->RunAsync(tensors);
      fu.get();
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  double time_span =
      std::chrono::duration<double, std::milli>(end - begin).count();

  tensors.clear();
  std::cout << "Run " << config.iterations << " inferences request on "
            << time_span << " ms, average inference time is "
            << time_span / config.iterations << " ms."
            << " latency is "<< latency_time / config.iterations  << " ms."
            << std::endl;

  std::cout << "=== Performance summary ===" << std::endl;
  std::cout << "Throughput: " << config.iterations / time_span * 1000 << " qps. "
            << "average time: " << time_span / config.iterations << " ms"
            << std::endl;
  std::cout << "latency: " << latency_time / config.iterations << " ms"
            << std::endl;


  ctx->Release();
  engine.Release();
}

};  // namespace TECO_INFER_NS
