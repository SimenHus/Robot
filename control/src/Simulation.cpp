#include "Simulation.h"


using namespace Simulation;

std::vector<Eigen::VectorXd> Simulation::LQR(const std::vector<float> &time, const LinearSystems::LQR &controller, const Eigen::VectorXd &x0, const std::vector<Eigen::VectorXd> &reference) {
    uint interpolationSteps = 100;

    std::vector<Eigen::VectorXd> result(time.size());
    result[0] = controller.observe(x0);

    Eigen::VectorXd x = x0;
    Eigen::VectorXd r;
    Eigen::VectorXd u;

    float dt = (time[1] - time[0]) / interpolationSteps;
    for (uint i = 1; i < time.size(); i++) {
        r = reference[i-1];
        for (uint j = 0; j < interpolationSteps; j++) {
            u = controller.calculateGain(x, r);
            x = x + controller.process(x, u)*dt;
        }

        result[i] = controller.observe(x, u);
    }

    return result;
}