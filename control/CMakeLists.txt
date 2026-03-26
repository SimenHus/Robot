add_library(control STATIC
    src/Linear.cpp
    src/Simulation.cpp
    src/NonLinear.cpp
)

find_package(Eigen3 REQUIRED)
target_link_libraries(control PRIVATE Eigen3::Eigen)

# Make include folder publicly available
target_include_directories(control PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)