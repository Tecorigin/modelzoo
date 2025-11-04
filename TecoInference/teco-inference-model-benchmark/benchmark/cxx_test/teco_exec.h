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

#include <teco_infer/common/common.h>
#include <teco_infer/core/device_type.h>
#include <teco_infer/core/teco_interface_api.h>
#include <teco_infer/utils/tensor/empty.h>
#include <teco_infer/utils/version_utils.h>
#include <teco_infer/core/version.h>
#include <teco_infer/core/macro.h>

#include <cctype>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "git_version.h"

struct TecoExecArgConfig {
  std::string engine_path;
  int warmup = 10;
  int iterations = 50;
  int depth = 2;
  std::vector<std::vector<int64_t>> shapes;
  std::vector<std::string> dtypes;
  std::string dtype = "fp32";
  bool run_async = true;
};

namespace TECO_INFER_NS {
using TimeClock = std::chrono::high_resolution_clock::time_point;

extern std::unordered_map<std::string, TECO_INFER_NS::ScalarType> teco_exec_dtype_map;

inline std::vector<std::string> ParseArgs(const std::string& str, const char& key) {
  std::stringstream ss(str);
  std::string token;
  std::vector<std::string> result;
  while (std::getline(ss, token, key)) {
    result.push_back(token);
  }
  return result;
}

inline TECO_INFER_NS::Tensor GenTensor(std::vector<int64_t>* input_shape,
                                const std::string& dtype) {
  TECO_INFER_NS::Tensor ret;
  auto ktype = TECO_INFER_NS::teco_exec_dtype_map[dtype];
  auto tensor_options = TECO_INFER_NS::TensorOptions("cpu").dtype(ktype);
  TECO_INFER_NS::TensorShape target_shape(
      static_cast<int64_t*>(input_shape->data()), input_shape->size());
  ret = empty(target_shape, tensor_options);
  auto size_byte = compute_bytes(target_shape, ktype);
  std::memset(ret.mutable_data<void>(), 0, size_byte);
  return ret;
}

inline std::vector<TECO_INFER_NS::Tensor> GetInputTensor(
    std::vector<std::vector<int64_t>>* input_shapes,
    const std::vector<std::string>& input_dtypes) {
  std::vector<TECO_INFER_NS::Tensor> tensors;
  if (!input_dtypes.empty()) {
    if (input_shapes->size() != input_dtypes.size()) {
      throw std::invalid_argument("input shapes size is: " + std::to_string(input_shapes->size()) +
                ", input dtype size is: " + std::to_string(input_dtypes.size()) + ", please check inputs.");
    }
    for (size_t i = 0; i < input_shapes->size(); ++i) {
      tensors.push_back(GenTensor(&((*input_shapes)[i]), input_dtypes[i]));
    }
  } else {
    for (size_t i = 0; i < input_shapes->size(); ++i) {
      tensors.push_back(GenTensor(&((*input_shapes)[i]), "fp16"));
    }
  }
  return tensors;
}

inline void PrintArgs() {
  std::cout << "=== Build Options ===" << std::endl;
  std::cout << "--loadEngine=<file>"
            << "    Load a serialized tecoInference engine. " << std::endl
            << std::endl;
  std::cout
      << "--fp16"
      << "    Enable fp16 precision, in addition to fp32(default = disable)"
      << std::endl
      << std::endl;
  std::cout << "=== Inference Options ===" << std::endl;
  std::cout << "--input_shapes=spec"
            << "    Set input shapes. " << std::endl
            << "    Example input shapes spec: "
               "--input_shapes=1x3x256x256,1x3x128x128. Each input shape"
            << std::endl
            << "    is supplied as a value where value is the dimensions "
               "(including the batch dimension)"
            << std::endl
            << "    to be used for that input. Multiple input shapes can be "
               "provided via comma-separated value."
            << std::endl
            << std::endl;
  std::cout << "--iterations=N"
            << "    Run at least N inference iterations (default = 50)"
            << std::endl
            << std::endl;
  std::cout << "--warmUp=N"
            << "    Run for N inference iterations to warmup before measure "
               "performance (default = 10)"
            << std::endl
            << std::endl;
  std::cout << "--runSync"
            << "    Enable run sync, in addition to async(default = disable)"
            << std::endl
            << std::endl;
  std::cout << "--input_dtype=spec"
            << "    Set input shapes dtype. Support type: bool, fp16, fp32, int32, int64"
            << std::endl
            << "    Example input shapes spec: --input_dtype=fp16. Each input "
               "shape dtype."
            << std::endl
            << "    If the dtype is set, number must be the same as the "
               "input_shapes number."
            << std::endl
            << "    Multiple input shapes dytpe can be provided via "
               "comma-separated value(default = fp16)."
            << std::endl
            << std::endl;
  std::cout << "--version"
            << "    Print the version of tecoexec." << std::endl
            << std::endl;
}

inline void show3rdParty() {
  std::cout << "---------------+-----------------------------------------------" << std::endl;
  std::cout << "Teco-Infer     | "
            << TECO_INFER_NS::normalizeComponentVersion(TECO_INFER_NS::kTecoInferVersion)
            << "+git"<< GIT_HEAD_VERSION
            << std::endl;
  std::cout << "TecoDNN        | "
            << TECO_INFER_NS::normalizeComponentVersion(TECO_INFER_NS::get_tecodnn_version())
            << " (" << TECO_INFER_NS::tecodnn_lib_path() << ")"
            << std::endl;
  std::cout << "TecoBLAS       | "
            << TECO_INFER_NS::normalizeComponentVersion(TECO_INFER_NS::get_tecoblas_version())
            << " (" << TECO_INFER_NS::tecoblas_lib_path() << ")"
            << std::endl;
  std::cout << "TecoCustom     | "
            << TECO_INFER_NS::normalizeComponentVersion(TECO_INFER_NS::get_tecocustom_version())
            << " (" << TECO_INFER_NS::customdnn_lib_path() << ")"
            << std::endl;
  std::cout << "SDAA Runtime   | "
            << TECO_INFER_NS::normalizeComponentVersion(TECO_INFER_NS::get_sdaa_runtime_version())
            << " (" << TECO_INFER_NS::sdaa_runtime_lib_path() << ")"
            << std::endl;
  std::cout << "SDAA Driver    | "
            << TECO_INFER_NS::normalizeComponentVersion(TECO_INFER_NS::get_sdaa_driver_version())
            << std::endl;
  std::cout << "---------------+-----------------------------------------------" << std::endl;
}

inline void Update(const std::string& str,
              std::function<void(const std::string&, TecoExecArgConfig*)> fn,
              std::map<std::string, std::function<void(const std::string&, TecoExecArgConfig*)>>* parse_maps) {
    std::string arg = "--" + str;
    (*parse_maps)[arg] = fn;
}

inline void TecoProcess(std::map<std::string, std::function<void(const std::string&, TecoExecArgConfig*)>>* parse_maps) {
  TECO_INFER_NS::Update("loadEngine", [](const std::string& path, TecoExecArgConfig* args) {
    args->engine_path = path;
  }, parse_maps);
  TECO_INFER_NS::Update("help",
               [](const std::string& str, TecoExecArgConfig* args) { PrintArgs(); }, parse_maps);
  TECO_INFER_NS::Update("fp16", [](const std::string& str, TecoExecArgConfig* args) {
    args->dtype = "fp16";
  }, parse_maps);
  TECO_INFER_NS::Update("warmUp", [](const std::string& str, TecoExecArgConfig* args) {
    args->warmup = std::stoi(str);
  }, parse_maps);
  TECO_INFER_NS::Update("iterations", [](const std::string& str, TecoExecArgConfig* args) {
    args->iterations = std::stoi(str);
  }, parse_maps);
  TECO_INFER_NS::Update("input_shapes", [](const std::string& str, TecoExecArgConfig* args) {
    auto inputs = ParseArgs(str, ',');
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto tmp = ParseArgs(inputs[i], '*');
      std::vector<int64_t> shape;
      for (auto iter : tmp) {
        for (auto c : iter) {
          if (!std::isdigit(c)) {
            std::string errormessage = "input shapes has invalid argument: "+ inputs[i] + ", please check shapes.";
            throw std::invalid_argument(errormessage);
          }
        }
        try {
          int64_t count = static_cast<int64_t>(std::stoi(iter));
          shape.push_back(count);
                } catch (std::out_of_range& e) {
          std::string errormessage = "input shapes Out of range: ";
          errormessage += iter;
          throw std::out_of_range(errormessage);
        }
      }
      args->shapes.push_back(shape);
    }
  }, parse_maps);
  TECO_INFER_NS::Update("runSync", [](const std::string& str, TecoExecArgConfig* args) {
    args->run_async = false;
  }, parse_maps);
  TECO_INFER_NS::Update("input_dtype", [](const std::string& str, TecoExecArgConfig* args) {
    auto inputs = ParseArgs(str, ',');
    for (size_t i = 0; i < inputs.size(); ++i) {
      std::string type = inputs[i];
      if (teco_exec_dtype_map.find(type) != teco_exec_dtype_map.end()) {
        args->dtypes.push_back(type);
      } else {
        throw std::invalid_argument("input shapes dtype is not support: " + type + ", please check input dtype. ");
      }
    }
  }, parse_maps);
  TECO_INFER_NS::Update("version", [](const std::string& str, TecoExecArgConfig* args) {
    std::cout << "Tecoexec version: "
              << TECO_INFER_NS::normalizeComponentVersion(TECO_INFER_NS::kTecoInferVersion)
              << "+git"<< GIT_HEAD_VERSION
              << std::endl;
  }, parse_maps);
}


class __attribute__((visibility("default"))) TecoExec {
 public:

  void Executor(const std::string& arg, TecoExecArgConfig* config) {
    auto result = ParseArgs(arg, '=');
    if (parse_maps_.find(result[0]) != parse_maps_.end()) {
      auto fn = parse_maps_[result[0]];
      fn(result[1], config);
    } else {
      parse_maps_.clear();
      throw std::invalid_argument(" methon is notUpdate support:" + result[0] + ", please check args.");
    }
  }



  void TecoRun(const TecoExecArgConfig& config);

  std::map<std::string, std::function<void(const std::string&, TecoExecArgConfig*)>>
      parse_maps_;
};

};// namespace TECO_INFER_NS
