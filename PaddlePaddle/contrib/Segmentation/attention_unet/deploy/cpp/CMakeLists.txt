cmake_minimum_required(VERSION 3.0)
project(cpp_inference_demo CXX C)

option(WITH_MKL        "Compile demo with MKL/OpenBlas support, default use MKL."       ON)
option(WITH_GPU        "Compile demo with GPU/CPU, default use CPU."                    OFF)
option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static."   ON)
option(USE_TENSORRT "Compile demo with TensorRT."   OFF)
option(WITH_ROCM "Compile demo with rocm." OFF)

if (WIN32)
  SET(PADDLE_LIB "" CACHE PATH "Location of libraries")
  SET(PADDLE_LIB_NAME "" CACHE STRING "libpaddle_inference")

  include(cmake/yaml-cpp.cmake)
  include_directories("${CMAKE_SOURCE_DIR}/")
  include_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/src/ext-yaml-cpp/include")
  link_directories("${CMAKE_CURRENT_BINARY_DIR}/ext/yaml-cpp/lib")
endif()

if(NOT WITH_STATIC_LIB)
  add_definitions("-DPADDLE_WITH_SHARED_LIB")
else()
  # PD_INFER_DECL is mainly used to set the dllimport/dllexport attribute in dynamic library mode.
  # Set it to empty in static library mode to avoid compilation issues.
  add_definitions("/DPD_INFER_DECL=")
endif()

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if(NOT DEFINED PADDLE_LIB)
  message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()
if(NOT DEFINED DEMO_NAME)
  message(FATAL_ERROR "please set DEMO_NAME with -DDEMO_NAME=demo_name")
endif()

include_directories("${PADDLE_LIB}/")
set(PADDLE_LIB_THIRD_PARTY_PATH "${PADDLE_LIB}/third_party/install/")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}cryptopp/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}onnxruntime/include")
include_directories("${PADDLE_LIB_THIRD_PARTY_PATH}paddle2onnx/include")

link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}protobuf/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}glog/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}gflags/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}xxhash/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}cryptopp/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}onnxruntime/lib")
link_directories("${PADDLE_LIB_THIRD_PARTY_PATH}paddle2onnx/lib")
link_directories("${PADDLE_LIB}/paddle/lib")

if (WIN32)
  add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
  option(MSVC_STATIC_CRT "use static C Runtime library by default" ON)
  if (MSVC_STATIC_CRT)
    if (WITH_MKL)
      set(FLAG_OPENMP "/openmp")
    endif()
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    safe_set_static_flag()
    if (WITH_STATIC_LIB)
      add_definitions(-DSTATIC_LIB)
    endif()
  endif()
else()
  if(WITH_MKL)
    set(FLAG_OPENMP "-fopenmp")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${FLAG_OPENMP}")
endif()

if(WITH_GPU)
  if(NOT WIN32)
    set(CUDA_LIB "/usr/local/cuda/lib64/" CACHE STRING "CUDA Library")
  endif(NOT WIN32)
endif()

if(NOT WIN32)
    if (USE_TENSORRT AND WITH_GPU)
      set(TENSORRT_ROOT "" CACHE STRING "The root directory of TensorRT library")
      if("${TENSORRT_ROOT}" STREQUAL "")
          message(FATAL_ERROR "The TENSORRT_ROOT is empty, you must assign it a value with CMake command. Such as: -DTENSORRT_ROOT=TENSORRT_ROOT_PATH ")
      endif()
      set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
      set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)
      file(READ ${TENSORRT_INCLUDE_DIR}/NvInfer.h TENSORRT_VERSION_FILE_CONTENTS)
      string(REGEX MATCH "define NV_TENSORRT_MAJOR +([0-9]+)" TENSORRT_MAJOR_VERSION
        "${TENSORRT_VERSION_FILE_CONTENTS}")
      if("${TENSORRT_MAJOR_VERSION}" STREQUAL "")
        file(READ ${TENSORRT_INCLUDE_DIR}/NvInferVersion.h TENSORRT_VERSION_FILE_CONTENTS)
        string(REGEX MATCH "define NV_TENSORRT_MAJOR +([0-9]+)" TENSORRT_MAJOR_VERSION
          "${TENSORRT_VERSION_FILE_CONTENTS}")
      endif()
      if("${TENSORRT_MAJOR_VERSION}" STREQUAL "")
        message(SEND_ERROR "Failed to detect TensorRT version.")
      endif()
      string(REGEX REPLACE "define NV_TENSORRT_MAJOR +([0-9]+)" "\\1"
        TENSORRT_MAJOR_VERSION "${TENSORRT_MAJOR_VERSION}")
      message(STATUS "Current TensorRT header is ${TENSORRT_INCLUDE_DIR}/NvInfer.h. "
        "Current TensorRT version is v${TENSORRT_MAJOR_VERSION}. ")
      include_directories("${TENSORRT_INCLUDE_DIR}")
      link_directories("${TENSORRT_LIB_DIR}")
    endif()
endif()

if(WITH_MKL)
  set(MATH_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mklml")
  include_directories("${MATH_LIB_PATH}/include")
  if(WIN32)
    set(MATH_LIB ${MATH_LIB_PATH}/lib/mklml${CMAKE_STATIC_LIBRARY_SUFFIX}
                 ${MATH_LIB_PATH}/lib/libiomp5md${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(MATH_LIB ${MATH_LIB_PATH}/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
                 ${MATH_LIB_PATH}/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
  set(MKLDNN_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}mkldnn")
  if(EXISTS ${MKLDNN_PATH})
    include_directories("${MKLDNN_PATH}/include")
    if(WIN32)
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
    else()
      set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libdnnl.so.3)
    endif()
  endif()
else()
  set(OPENBLAS_LIB_PATH "${PADDLE_LIB_THIRD_PARTY_PATH}openblas")
  include_directories("${OPENBLAS_LIB_PATH}/include/openblas")
  if(WIN32)
    set(MATH_LIB ${OPENBLAS_LIB_PATH}/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(MATH_LIB ${OPENBLAS_LIB_PATH}/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
endif()

if(WITH_STATIC_LIB)
  if(WIN32)
    set(DEPS ${PADDLE_LIB}/paddle/lib/${PADDLE_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
  endif()
else()
  if(WIN32)
    set(DEPS ${PADDLE_LIB}/paddle/lib/${PADDLE_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX})
  else()
    set(DEPS ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
endif()

if (NOT WIN32)
  set(EXTERNAL_LIB "-lrt -ldl -lpthread")
  set(DEPS ${DEPS}
      ${MATH_LIB} ${MKLDNN_LIB}
      glog gflags protobuf xxhash cryptopp
      ${EXTERNAL_LIB})
else()
  set(DEPS ${DEPS}
      ${MATH_LIB} ${MKLDNN_LIB}
      glog gflags_static libprotobuf xxhash cryptopp-static libyaml-cppmt ${EXTERNAL_LIB})
	  set(DEPS ${DEPS} libcmt shlwapi)
endif(NOT WIN32)

if(WITH_GPU)
  if(NOT WIN32)
    if (USE_TENSORRT)
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${TENSORRT_LIB_DIR}/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    SET(CUDA_LIB "" CACHE PATH "Location of libraries")
    if (USE_TENSORRT)
      set(DEPS ${DEPS} ${CUDA_LIB}/nvinfer${CMAKE_STATIC_LIBRARY_SUFFIX})
      set(DEPS ${DEPS} ${CUDA_LIB}/nvinfer_plugin${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
    set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
    set(DEPS ${DEPS} ${CUDA_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX} )
  endif()
endif()

if(WITH_ROCM)
  if(NOT WIN32)
    set(DEPS ${DEPS} ${ROCM_LIB}/libamdhip64${CMAKE_SHARED_LIBRARY_SUFFIX})
  endif()
endif()

if(NOT WIN32)
  include_directories(/usr/local/include)
  link_directories(/usr/local/lib)

  find_package(yaml-cpp REQUIRED)
  include_directories(${YAML_CPP_INCLUDE_DIRS})
  link_directories(${YAML_CPP_LIBRARIES})
  set(DEPS ${DEPS} "-lyaml-cpp")

  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  set(DEPS ${DEPS} ${OpenCV_LIBS})
	
  add_executable(${DEMO_NAME} src/${DEMO_NAME}.cc)
  target_link_libraries(${DEMO_NAME} ${DEPS})
else()
  include_directories("${PADDLE_LIB}/paddle/fluid/inference")
  include_directories("${PADDLE_LIB}/paddle/include")
  link_directories("${PADDLE_LIB}/paddle/fluid/inference")
  
  SET(OPENCV_DIR "" CACHE PATH "Location of libraries")
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/build/ NO_DEFAULT_PATH)
  include_directories(${OpenCV_INCLUDE_DIRS})
  set(DEPS ${DEPS} ${OpenCV_LIBS})
	
  add_executable(${DEMO_NAME} src/${DEMO_NAME}.cc)
  ADD_DEPENDENCIES(${DEMO_NAME} ext-yaml-cpp)
  message("DEPS:" ${DEPS})
  target_link_libraries(${DEMO_NAME} ${DEPS})
endif()

if(WIN32 AND USE_TENSORRT)
  SET(TENSORRT_DLL "" CACHE PATH "Location of TensorRT .dll")
  add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORRT_DLL}/nvinfer.dll ./nvinfer.dll
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORRT_DLL}/nvinfer.dll ./release/nvinfer.dll
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORRT_DLL}/nvinfer_plugin.dll ./nvinfer_plugin.dll
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TENSORRT_DLL}/nvinfer_plugin.dll ./release/nvinfer_plugin.dll
  )
endif()

if(WIN32 AND WITH_MKL)
  add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.dll ./mklml.dll
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.dll ./libiomp5md.dll
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mkldnn/lib/mkldnn.dll ./mkldnn.dll
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.dll ./release/mklml.dll
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.dll ./release/libiomp5md.dll
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/mkldnn/lib/mkldnn.dll ./release/mkldnn.dll
  )
endif()

if(WIN32 AND NOT WITH_MKL)
  add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/openblas/lib/openblas.dll ./openblas.dll
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/openblas/lib/openblas.dll ./release/openblas.dll
  )
endif()

if (WIN32)
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/onnxruntime/lib/onnxruntime.dll ./onnxruntime.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/paddle2onnx/lib/paddle2onnx.dll ./paddle2onnx.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/onnxruntime/lib/onnxruntime.dll ./release/onnxruntime.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/third_party/install/paddle2onnx/lib/paddle2onnx.dll ./release/paddle2onnx.dll
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PADDLE_LIB}/paddle/lib/${PADDLE_LIB_NAME}.dll ./release/${PADDLE_LIB_NAME}.dll
    )
endif()
