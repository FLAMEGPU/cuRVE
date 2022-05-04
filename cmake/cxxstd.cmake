# Require CXX 17, this is probably not actually required for cuRVE, but matches the use-case in FLAME GPU 2 
set(FLAMEGPU_CXX_STD 17)
# @future - set this on a per target basis using set_target_properties?
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD ${FLAMEGPU_CXX_STD})
    set(CMAKE_CXX_STANDARD_REQUIRED true)
endif()
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD ${FLAMEGPU_CXX_STD})
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()