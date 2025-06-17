cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_library(cpp182 STATIC ${SRC_FILES})

file(GLOB TEST_FILES ${CMAKE_SOURCE_DIR}/tests/*.cpp)
foreach(test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} cpp182)
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -O2)
endforeach()

file(GLOB TEST_FILES ${CMAKE_SOURCE_DIR}/experiments/*.cpp)
foreach(test_file ${TEST_FILES})
    get_filename_component(test_name ${test_file} NAME_WE)
    add_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} cpp182)
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -O2)
endforeach()

add_executable(main main.cpp)
target_link_libraries(main cpp182)
target_compile_options(main PRIVATE -Wall -Wextra -O2)