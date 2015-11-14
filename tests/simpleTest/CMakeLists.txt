set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})

add_executable(simpleTest src/simpleTest.cpp)
target_link_libraries(simpleTest PUBLIC libsvmpp)

add_test(simpleTest1 simpleTest)