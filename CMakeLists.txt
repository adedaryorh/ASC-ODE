cmake_minimum_required(VERSION 3.20)
project(ASC_ode)

set (CMAKE_CXX_STANDARD 20)


include_directories(src nanoblas/src)

add_subdirectory (nanoblas)

add_subdirectory (mechsystem)

add_executable (test_ode demos/test_ode.cpp)
target_link_libraries (test_ode PUBLIC nanoblas)

add_executable (demo_autodiff demos/demo_autodiff.cpp)
target_link_libraries (demo_autodiff PUBLIC nanoblas)


add_executable(ch17 exercises/ch17.cpp)
option(ASC_EXERCISE_TESTS "Build Catch2 v2 exercise tests" OFF)
if(ASC_EXERCISE_TESTS)
  enable_testing()
  find_package(Catch2 2 REQUIRED)
  add_executable(ode_orders tests/orders.cpp)
  target_link_libraries(ode_orders PRIVATE Catch2::Catch2)
  add_test(NAME ode_orders COMMAND ode_orders)
endif()
add_executable(ch18 exercises/ch18.cpp)
add_executable(ch19 exercises/ch19.cpp)
add_executable(ch20 exercises/ch20.cpp)
add_executable(ch21 exercises/ch21.cpp)
