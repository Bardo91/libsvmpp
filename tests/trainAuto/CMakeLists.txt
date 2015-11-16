cmake_minimum_required (VERSION 2.8)
project(libsvmpp)

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})

add_executable(trainAuto src/trainAuto.cpp)
target_link_libraries(trainAuto PUBLIC libsvmpp)

add_test(trainAuto1 trainAuto)