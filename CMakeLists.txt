cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(main 
    main.cpp
    src/dataloader.cpp
)

include_directories(${CMAKE_SOURCE_DIR}/include)

target_compile_options(main PRIVATE -Wall -Wextra -O2)