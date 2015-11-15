cmake_minimum_required (VERSION 2.6)
if(POLICY CMP0048)
	cmake_policy(SET CMP0048 OLD)
endif()

project(libsvmpp)
set(PROJECT_VERSION "alpha_1.0.0")
enable_testing()

# Project directory
add_subdirectory(src)

# Dependencies
add_subdirectory(tests)
