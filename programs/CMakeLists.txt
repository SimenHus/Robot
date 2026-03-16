
add_executable(ControlTest ControlTest.cpp)
add_executable(EigenTest EigenTest.cpp)

target_link_libraries(ControlTest PRIVATE Eigen3::Eigen control)
target_link_libraries(EigenTest PRIVATE Eigen3::Eigen)
