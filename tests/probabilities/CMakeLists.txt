cmake_minimum_required (VERSION 2.8)
project(libsvmpp)

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})

add_executable(probabilities src/probabilities.cpp)
target_link_libraries(probabilities PUBLIC libsvmpp)

add_test(probabilities1 probabilities)