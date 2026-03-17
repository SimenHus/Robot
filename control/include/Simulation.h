#pragma once
#include <Eigen/Dense>
#include <vector>

#include "Linear.h"



namespace Simulation {

    std::vector<Eigen::VectorXd> LQR(const LinearSystems::StateSpace &sys, const LinearSystems::CostMatrices &cost,
    const Eigen::VectorXd &x0, const uint iterations = 1e3, const float dt = 1e-3);

};