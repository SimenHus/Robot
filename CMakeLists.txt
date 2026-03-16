# Define minimum CMAKE version and set current project version
cmake_minimum_required(VERSION 3.20)
project(Robot VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Add subdirectories. Needed to include libraries
add_subdirectory(programs)
add_subdirectory(control)

# Add external directories
add_subdirectory(external/eigen)

# Copy all assets to the build folder for easier development
file(COPY ${CMAKE_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})
