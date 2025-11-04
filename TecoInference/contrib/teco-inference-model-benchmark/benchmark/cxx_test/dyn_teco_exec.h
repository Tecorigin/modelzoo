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
#pragma once

#include "teco_infer_dyn/interface/common.h"
#include <teco_infer_dyn/interface/future.h>
#include <teco_infer_dyn/interface/teco_interface_api.h>
#include <teco_infer_dyn/interface/tensor.h>
#include <teco_infer_dyn/interface/utils.h>
#include "teco_exec.h"
// #include <teco_infer/utils/version_utils.h>
// #include <teco_infer/core/version.h>
// #include <teco_infer/core/macro.h>

#include <cctype>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <sstream>
#include <functional>


// #include "git_version.h"

namespace TECO_INFER {

using TimeClock = std::chrono::high_resolution_clock::time_point;

extern std::unordered_map<std::string, TECO_INFER::IScalarType> teco_exec_dyn_dtype_map;

inline std::vector<TECO_INFER::IScalarType> GetInputDtype(const std::vector<std::string>& input_dtypes){
  std::vector<TECO_INFER::IScalarType> ret;
  for (auto iter : input_dtypes) {
    if (teco_exec_dyn_dtype_map.find(iter) != teco_exec_dyn_dtype_map.end()) {
        ret.push_back(teco_exec_dyn_dtype_map[iter]);
    } else {
        return ret;
    }
  }
  return ret;
}

inline std::vector<TECO_INFER::Tensor> GetInputTensor(
    std::vector<std::vector<int64_t>>* input_shapes,
    const std::vector<std::string>& input_dtypes) {
  std::vector<TECO_INFER::Tensor> tensors;
  if (!input_dtypes.empty()) {
    if (input_shapes->size() != input_dtypes.size()) {
      throw std::invalid_argument("input shapes size is: " + std::to_string(input_shapes->size()) +
                ", input dtype size is: " + std::to_string(input_dtypes.size()) + ", please check inputs.");
    }
    auto dtypes = GetInputDtype(input_dtypes);
    tensors =  TECO_INFER::CreateTensor(*input_shapes, dtypes);
    
  } else {
    for (size_t i = 0; i < input_shapes->size(); ++i) {
      std::vector<TECO_INFER::IScalarType> dytpes;
      for (size_t i = 0; i < input_shapes->size(); ++i) {
        dytpes.push_back(TECO_INFER::IScalarType::Half);
      }
      tensors = TECO_INFER::CreateTensor(*input_shapes, dytpes);
    }
  }
  return tensors;
}


class __attribute__((visibility("default"))) TecoDynExec {
 public:

  void Executor(const std::string& arg, TecoExecArgConfig* config);

  void TecoRun(const TecoExecArgConfig& config);

  std::map<std::string, std::function<void(const std::string&, TecoExecArgConfig*)>> parse_maps_;
};

};// namespace TECO_INFER_NS
