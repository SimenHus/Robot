#include "Simulation.h"



using namespace Simulation;


std::vector<Eigen::VectorXd> Simulation::LQR(const LinearSystems::StateSpace &sys, const LinearSystems::CostMatrices &cost,
    const Eigen::VectorXd &x0, const uint iterations, const float dt) {
    
    std::vector<Eigen::VectorXd> result(iterations);
    result[0] = x0;

    auto [K, P] = LinearSystems::LQR(sys, cost);
    
    Eigen::VectorXd x_next;
    for (int i = 1; i < iterations; i++) {
        x_next = result[i-1] + (sys.A - sys.B*K)*result[i-1] * dt;
        result[i] = x_next;
    }


    return result;
}