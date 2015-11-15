cmake_minimum_required (VERSION 2.8)
project(libsvmpp)

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})

add_executable(rbfKernel src/rbfKernel.cpp)
target_link_libraries(rbfKernel PUBLIC libsvmpp)

add_test(rbfKernel1 rbfKernel)