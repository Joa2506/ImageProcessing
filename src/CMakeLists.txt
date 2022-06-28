# src/CMakeLists.txt


# set the project name
project("Image processor")
cmake_minimum_required(VERSION 3.10)

#links libraries
#target_link_libraries(${TARGET_NAME} nvonnxparser)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(TensorRT_DIR "/usr/src/tensorrt")
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package( OpenCV 4.5.5 REQUIRED )
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")
message(${OpenCV_DIR})
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

if(CUDA_VERSION_MAJOR GREATER 9)
	message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_86")

	set(
		CUDA_NVCC_FLAGS
		${CUDA_NVCC_FLAGS}
		-gencode arch=compute_86,code=sm_86 #Uncomment for 8.6 architectures
	)
endif()


# specify the executable and corresponding source file
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})  #This seems to have added the cuda_runtime api.
include_directories(${TensorRT_INCLUDE_DIR})
include_directories(${TensorRT_DIR}/samples/common)
include_directories(${OpenCV_INCLUDE_DIRS})
cuda_add_executable(run main.cpp Engine.cpp Engine.cu ImageProcessing.cu)


#links libraries
target_link_libraries(run ${CUDA_LIBRARIES})
target_link_libraries(run ${OpenCV_LIBS})

# cmake_minimum_required(VERSION 3.10)

# project( imageproc )
# find_package( OpenCV REQUIRED )
# find_package(CUDA REQUIRED)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86)

# cuda_add_executable( run main.cpp Engine.cpp Engine.cu)
# target_link_libraries( run ${OpenCV_LIBS} )