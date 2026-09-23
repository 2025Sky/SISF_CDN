# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Ensure C++17 is enforced

include(FetchContent)
include(CMakePrintHelpers)

include_directories(./ffmpeg_HDF5_filter/src)

# Define the project name
project(nTracer_cdn)

# Add an executable target
add_executable(nTracer_cdn src/app.cpp)

target_link_libraries(nTracer_cdn pthread)
target_link_libraries(nTracer_cdn sqlite3)

target_link_libraries(nTracer_cdn avcodec)
target_link_libraries(nTracer_cdn avformat)
target_link_libraries(nTracer_cdn avutil)
target_link_libraries(nTracer_cdn swscale)

find_library(ZSTD_LIB zstd PATH ./zstd/lib/)
target_link_libraries(nTracer_cdn "${ZSTD_LIB}")
cmake_print_variables(ZSTD_LIB)

find_library(X264_LIB x264 PATH ./x264/)
target_link_libraries(nTracer_cdn "${X264_LIB}")
cmake_print_variables(X264_LIB)

find_library(CAI_FFMPEG_LIB h5ffmpeg_shared PATH ./ffmpeg_HDF5_filter/)
target_link_libraries(nTracer_cdn "${CAI_FFMPEG_LIB}")
cmake_print_variables(CAI_FFMPEG_LIB)

target_include_directories(nTracer_cdn PRIVATE "/usr/include/hdf5/serial/")

# -march=native tied the binary to whichever CPU built it; CI runners vary
# (AMD Zen 3 on one run, Intel Sapphire Rapids on the next), and a binary
# built for the latter may use instructions the production host lacks.
# x86-64-v3 (AVX2) runs on every current runner and on the production host.
# Other architectures (local arm64 builds) keep native.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(NTRACER_MARCH "x86-64-v3" CACHE STRING "Target CPU for nTracer_cdn")
else()
  set(NTRACER_MARCH "native" CACHE STRING "Target CPU for nTracer_cdn")
endif()
target_compile_options(nTracer_cdn PUBLIC -O3 -march=${NTRACER_MARCH} -Wl,--no-as-needed -ldl)

# AddressSanitizer + UBSan for this target only (tensorstore and grpc stay
# uninstrumented). The runtimes are linked statically so the slim runtime
# image needs no extra packages.
option(NTRACER_SANITIZE "Build nTracer_cdn with -fsanitize=address,undefined" OFF)
if(NTRACER_SANITIZE)
  target_sources(nTracer_cdn PRIVATE src/sanitizer_options.cpp)
  target_compile_options(nTracer_cdn PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
  target_link_options(nTracer_cdn PRIVATE -fsanitize=address,undefined -static-libasan -static-libubsan)
endif()


FetchContent_Declare(
  tensorstore
  URL "https://github.com/google/tensorstore/archive/refs/tags/v0.1.71.tar.gz"
)

FetchContent_MakeAvailable(tensorstore)


# Define a target that depends on TensorStore...
target_link_libraries(
  nTracer_cdn
#  PRIVATE
    tensorstore::tensorstore tensorstore::all_drivers
)
