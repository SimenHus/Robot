
add_executable(ControlTest ControlTest.cpp)

target_link_libraries(ControlTest PRIVATE Eigen3::Eigen control)
