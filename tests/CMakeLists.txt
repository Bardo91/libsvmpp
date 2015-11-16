cmake_minimum_required (VERSION 2.8)
project(libsvmpp)

# Project directory
add_subdirectory(linearKernel)
add_subdirectory(rbfKernel)
add_subdirectory(probabilities)
add_subdirectory(persistency)
add_subdirectory(trainAuto)