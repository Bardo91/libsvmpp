cmake_minimum_required (VERSION 2.8)
project(libsvmpp)

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})

add_executable(persistency src/persistency.cpp)
target_link_libraries(persistency PUBLIC libsvmpp)

add_test(persistency1 persistency)