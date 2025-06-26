cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang++-17)

include_directories(${CMAKE_SOURCE_DIR}/include)
#include_directories(/usr/local/cuda/include)
find_package(CUDAToolkit REQUIRED)


file(GLOB_RECURSE SRC_FILES
     CONFIGURE_DEPENDS
     ${CMAKE_SOURCE_DIR}/src/*.cpp
     ${CMAKE_SOURCE_DIR}/src/*.cu)

add_library(lib STATIC ${SRC_FILES})

#target_link_libraries(lib cudart cuda)
target_link_libraries(lib CUDA::cudart)

set_target_properties(lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

file(GLOB TEST_FILES ${CMAKE_SOURCE_DIR}/tests/*.cpp)
foreach(test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    set_target_properties(${test_name} PROPERTIES LINKER_LANGUAGE CUDA)
    target_link_libraries(${test_name} lib c++ c++abi)
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -O2)
endforeach()

file(GLOB EXP_FILES ${CMAKE_SOURCE_DIR}/experiments/*.cpp)
foreach(exp_file ${EXP_FILES})
    get_filename_component(exp_name ${exp_file} NAME_WE)
    add_executable(${exp_name} ${exp_file})
    set_target_properties(${exp_name} PROPERTIES LINKER_LANGUAGE CUDA)
    target_link_libraries(${exp_name} lib c++ c++abi)
    target_compile_options(${exp_name} PRIVATE -Wall -Wextra -O2)
    # target_compile_options(${exp_name} PRIVATE -Wall -Wextra -O3 -Rpass=inline -Rpass-missed=inline)
endforeach()

add_executable(main main.cpp)
set_target_properties(main PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(main lib c++ c++abi)
target_compile_options(main PRIVATE -Wall -Wextra -O2)