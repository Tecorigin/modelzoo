#include "teco_exec.h"
#include "dyn_teco_exec.h"

#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    TECO_INFER_NS::show3rdParty();
    const char* path = getenv("TECO_INFER_TVM_RUN");  
    TecoExecArgConfig config;
    std::string env_val;
    if (path != nullptr) {
      env_val = path;
      std::cout << "TECO_INFER_TVM_RUN : " << path << std::endl;
    }
    if (!env_val.empty() && (env_val == "1" || env_val == "true"|| env_val == "yes")) {
      std::cout << " Use TVM Runtime get profiler message. "<<std::endl;
      TECO_INFER_NS::TecoExec teco;
      TECO_INFER_NS::TecoProcess(&(teco.parse_maps_));
      for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        teco.Executor(arg, &config);
      }
      if (config.engine_path == "") {
        std::cout << " Engine path is empty, please set engine path. ";
        return 0;
      }
      teco.TecoRun(config);
    } else {
      std::cout << " Use DYN Runtime get profiler message. "<<std::endl;
      TECO_INFER::TecoDynExec teco;
      TECO_INFER_NS::TecoProcess(&(teco.parse_maps_));
      for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        teco.Executor(arg, &config);
      }
      if (config.engine_path == "") {
        std::cout << " Engine path is empty, please set engine path. ";
        return 0;
      }
      teco.TecoRun(config);
    }
    return 0;
}

