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
# include all cpp files in the tests exec except src/main.cpp
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS src/*.cpp test/*.cpp)
list(FILTER TEST_SOURCES EXCLUDE REGEX "src/main.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})
add_executable(${TESTS_NAME} ${TEST_SOURCES})

target_include_directories(${PROJECT_NAME}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${TESTS_NAME}
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(LIBS_TO_ALWAYS_LINK fmt::fmt Boost::multi_array Boost::qvm)
set(MAIN_LIBS_TO_LINK ${LIBS_TO_ALWAYS_LINK})
set(TEST_LIBS_TO_LINK ${LIBS_TO_ALWAYS_LINK} Matplot++::matplot Matplot++::cimg
                      doctest::doctest)

target_link_libraries(${PROJECT_NAME} PRIVATE ${MAIN_LIBS_TO_LINK})
target_link_libraries(${TESTS_NAME} PRIVATE ${TEST_LIBS_TO_LINK})
