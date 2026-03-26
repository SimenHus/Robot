
add_executable(ControlTest ControlTest.cpp)
add_executable(ParticleTest ParticleTest.cpp)

target_link_libraries(ControlTest PRIVATE Eigen3::Eigen control)
target_link_libraries(ParticleTest PRIVATE Eigen3::Eigen control)
