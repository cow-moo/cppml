cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX CUDA)

# C++ and CUDA versions
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Detect Clang toolchain and configure accordingly
if(APPLE)
    # macOS: Clang + libc++
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
else()
    # Linux: Clang + libstdc++
    set(CMAKE_CXX_COMPILER /usr/bin/clang++-17)
    set(CMAKE_C_COMPILER /usr/bin/clang-17)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang++-17)

    # Do NOT use libc++ on Linux — CUDA forbids it
    # Remove any -stdlib=libc++ flags
endif()

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
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -diag-suppress=3288 -diag-suppress=445>
)

# Helper function to define executables
function(add_my_executable target file)
    add_executable(${target} ${file})
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CUDA)
    target_link_libraries(${target} lib)

    target_compile_options(${target} PRIVATE -Wall -Wextra -O2)

    if(APPLE)
        target_compile_options(${target} PRIVATE -stdlib=libc++)
        target_link_options(${target} PRIVATE -stdlib=libc++)
    endif()
endfunction()

# Tests
file(GLOB TEST_FILES ${CMAKE_SOURCE_DIR}/tests/*.cpp)
foreach(test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_my_executable(${test_name} ${test_file})
endforeach()

# Experiments
file(GLOB EXP_FILES ${CMAKE_SOURCE_DIR}/experiments/*.cpp)
foreach(exp_file ${EXP_FILES})
    get_filename_component(exp_name ${exp_file} NAME_WE)
    add_my_executable(${exp_name} ${exp_file})
endforeach()

# Main
add_my_executable(main ${CMAKE_SOURCE_DIR}/main.cpp)