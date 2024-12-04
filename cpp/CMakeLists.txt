cmake_minimum_required(VERSION 3.15)

project(
  fmm-solver
  VERSION 0.1.0
  LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(fmt CONFIG REQUIRED)
find_package(Boost REQUIRED CONFIG COMPONENTS multi_array)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt Boost::multi_array)
