cmake_minimum_required(VERSION 3.15)

project(
  fmm-solver
  VERSION 0.1.0
  LANGUAGES CXX)
set(TESTS_NAME ${PROJECT_NAME}-test)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(fmt CONFIG REQUIRED)
find_package(doctest CONFIG REQUIRED)
find_package(Matplot++ CONFIG REQUIRED)
find_package(Boost REQUIRED CONFIG COMPONENTS multi_array qvm)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS test/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER include/fmm.hpp)

add_executable(${TESTS_NAME} ${TEST_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_include_directories(${TESTS_NAME} PRIVATE include)

set(MAIN_LIBS_TO_LINK fmt::fmt Boost::multi_array Boost::qvm)
set(TEST_LIBS_TO_LINK Matplot++::matplot Matplot++::cimg doctest::doctest
                      fmt::fmt ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE ${MAIN_LIBS_TO_LINK})
target_link_libraries(${TESTS_NAME} PRIVATE ${TEST_LIBS_TO_LINK})
