#pragma once
#include <Eigen/Dense>
#include <vector>
#include <iostream>

#include "Linear.h"


namespace Simulation {


std::vector<Eigen::VectorXd> LQR(const std::vector<float> &time, const LinearSystems::LQR &controller, const Eigen::VectorXd &x0, const std::vector<Eigen::VectorXd> &reference);



};