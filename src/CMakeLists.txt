cmake_minimum_required (VERSION 2.8)
project(libsvmpp)

# Collect all sources
file(GLOB_RECURSE LIBSVMPP_SOURCE_FILES "libsvmpp/*.cpp" "libsvmpp/*.h" "libsvmpp/*.inl" "libsvm/svm.h" "libsvm/svm.cpp" )

# Compiler specific options
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
endif()

# Add Executable
add_library(libsvmpp ${LIBSVMPP_SOURCE_FILES})
target_include_directories(libsvmpp PUBLIC ${PROJECT_SOURCE_DIR}/libsvmpp)
target_include_directories(libsvmpp PUBLIC ${PROJECT_SOURCE_DIR}/libsvm)

