project(hello_cnn)

cmake_minimum_required(VERSION 2.8)
SET( CMAKE_VERBOSE_MAKEFILE on )

add_compile_options(-std=c++11)

set(CMAKE_CXX_FLAGS "-Wall -O2")

# package
find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# source directories
aux_source_directory(. DIR_SRCS)

add_executable(cnn_test ${DIR_SRCS})

target_link_libraries(cnn_test ${OpenCV_LIBS})
