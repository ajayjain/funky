# Funky (Face)

## Dependencies
Eigen, OpenCV, dlib

## Building:
```
mkdir build
cd build
CC=gcc CXX=g++ cmake .. -DUSE_AVX_INSTRUCTIONS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build . --config Release -- -j `nproc`
```

Substitute CMAKE_BUILD_TYPE with "Release" or "Debug" as appropriate, and the build config with "Debug" if desired. Options for vectorization: `USE_AVX_INSTRUCTIONS`, `USE_SSE2_INSTRUCTIONS`, `USE_SSE4_INSTRUCTIONS`.
