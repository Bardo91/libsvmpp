cmake_minimum_required (VERSION 2.8)
project(libsvmpp)

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})

add_executable(linearKernel src/linearKernel.cpp)
target_link_libraries(linearKernel PUBLIC libsvmpp)

add_test(linearKernel1 linearKernel)