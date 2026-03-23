#pragma once
#include <vector>
#include <iostream>

#include "Linear.h"


namespace Simulation {


std::vector<Eigen::VectorXd> circularPath(const uint &iterations, const float &radius, const Eigen::VectorXd &start);

std::vector<Eigen::VectorXd> LQR(const std::vector<float> &time, const LinearSystems::LQR &controller, const Eigen::VectorXd &x0, const std::vector<Eigen::VectorXd> &reference);
std::vector<Eigen::VectorXd> KalmanFilter(const std::vector<Eigen::VectorXd> &groundTruth, const float &disturbance, LinearSystems::KalmanFilter &filter);


};