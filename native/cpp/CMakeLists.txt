# File: Builds the warning-clean C++ inference library, CLI, unit tests, and optional sanitizers.
cmake_minimum_required(VERSION 3.20)
project(edgevision_cpp VERSION 1.0.0 LANGUAGES CXX)

option(EDGEVISION_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

add_library(edgevision_model src/model.cpp)
target_include_directories(edgevision_model PUBLIC include)
target_compile_features(edgevision_model PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(edgevision_model PRIVATE /W4 /WX /permissive-)
else()
  target_compile_options(edgevision_model PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
  if(EDGEVISION_SANITIZERS)
    target_compile_options(edgevision_model PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(edgevision_model PUBLIC -fsanitize=address,undefined)
  endif()
endif()

add_executable(edgevision_cpp src/main.cpp)
target_link_libraries(edgevision_cpp PRIVATE edgevision_model)

include(CTest)
if(BUILD_TESTING)
  add_executable(edgevision_cpp_tests tests/model_tests.cpp)
  target_link_libraries(edgevision_cpp_tests PRIVATE edgevision_model)
  add_test(NAME edgevision_cpp_tests COMMAND edgevision_cpp_tests)
  set_tests_properties(
    edgevision_cpp_tests
    PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  )
endif()
