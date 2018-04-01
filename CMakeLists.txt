cmake_minimum_required(VERSION 2.8)
project( PointSetGeodesicShooting_CUDA )

# import library
SET(VTK_DIR "/home/cwang/Libraries/lib/cmake/vtk-7.1" CACHE PATH "VTK directory override" FORCE)
SET(ITK_DIR "/home/cwang/Libraries/lib/cmake/ITK-4.11" CACHE PATH "ITK directory override" FORCE)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# CUDA version of LBGFS
# find_package(CudaLBFGS REQUIRED)
# include_directories(${CUDALBDFS_INCLUDE_DIRS})

include_directories(../common/inc/)
include_directories(./include/)

# An online macro found on https://github.com/PointCloudLibrary/pcl/issues/776 
# that solves the VTK flag fucking with NVCC problem - works like a charm
get_directory_property(dir_defs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)
set(vtk_flags)
foreach(it ${dir_defs})
    if(it MATCHES "vtk*")
    list(APPEND vtk_flags ${it})
    endif()
endforeach()

foreach(d ${vtk_flags})
    remove_definitions(-D${d})
endforeach()

# NVCC config
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50; -lcufft;--std=c++11)

# Compile cuda objects
CUDA_ADD_LIBRARY(cuda_obj STATIC 
                 src/reducer.cu
                 src/hamiltonian_kernel.cu
                 src/hamiltonian.cu
                 src/Float2DVec.cu
                 src/ReadWriteVTK.cxx
                 src/GreedyParameters.cxx
                 src/PointSetShootingProblem.cu
                 src/check.cxx
                 )

CUDA_ADD_CUBLAS_TO_TARGET(cuda_obj)

# Link ITK/VTK to the cuda object
target_link_libraries(cuda_obj ${ITK_LIBRARIES})
target_link_libraries(cuda_obj ${VTK_LIBRARIES})

# Compiler config - make sure this comes after the nvcc compile to prevent interference to nvcc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -lpthread")

add_executable(lmshoot_cuda src/PointSetGeodesicShooting_CUDA.cpp) # only used one c++ class for this test

# Link ITK/VTK to executable 
target_link_libraries(lmshoot_cuda ${ITK_LIBRARIES} )
target_link_libraries(lmshoot_cuda ${VTK_LIBRARIES})

# Link CUDA to main
target_link_libraries(lmshoot_cuda ${CUDA_LIBRARIES})

# Link CUDA obj
target_link_libraries(lmshoot_cuda cuda_obj)

add_executable(shoot_forward src/ShootForward.cpp) # only used one c++ class for this test

# Link ITK/VTK to executable 
target_link_libraries(shoot_forward ${ITK_LIBRARIES} )
target_link_libraries(shoot_forward ${VTK_LIBRARIES})

# Link CUDA to main
target_link_libraries(shoot_forward ${CUDA_LIBRARIES})

# Link CUDA obj
target_link_libraries(shoot_forward cuda_obj)






