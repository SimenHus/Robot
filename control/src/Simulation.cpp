#include "Simulation.h"


using namespace Simulation;



std::vector<Eigen::VectorXd> Simulation::circularPath(const uint &iterations, const float &radius, const Eigen::VectorXd &start) {

    Eigen::VectorXd offset(2); offset << radius, 0;
    Eigen::VectorXd circleOrigin = start - offset;
    std::vector<Eigen::VectorXd> trajectory(iterations);
    const float dt = 2*3.14 / iterations;
    for (int i = 0; i < iterations; i++) {
        double theta = i * dt;  // angle increases over time

        double x = radius * std::cos(theta);
        double y = radius * std::sin(theta);

        Eigen::VectorXd position(2); position << x, y;
        trajectory[i] = circleOrigin + position;
    }
    return trajectory;
}

std::vector<Eigen::VectorXd> Simulation::LQR(const std::vector<float> &time, const LinearSystems::LQR &controller, const Eigen::VectorXd &x0, const std::vector<Eigen::VectorXd> &reference) {
    uint interpolationSteps = 100;

    std::vector<Eigen::VectorXd> result(time.size());
    result[0] = controller.measure(x0);

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

        result[i] = controller.measure(x, u);
    }

    return result;
}


std::vector<Eigen::VectorXd> Simulation::KalmanFilter(const std::vector<Eigen::VectorXd> &groundTruth, const float &disturbance, const LinearSystems::StateSpace &sys, const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0) {

    std::vector<Eigen::VectorXd> result(groundTruth.size());
    result[0] = x0;

    Eigen::MatrixXd P_prev = P0;
    for (uint i = 1; i < groundTruth.size(); i++) {
        Eigen::VectorXd measurement = groundTruth[i] + Eigen::VectorXd::Random(groundTruth[i].size()) * disturbance;
        auto [x_priori, P_priori] = LinearSystems::KalmanFilter::predict(sys, result[i-1], P_prev);
        auto [x_posterior, P_posterior] = LinearSystems::KalmanFilter::update(sys, x_priori, P_priori, measurement);
        result[i] = x_posterior;
        P_prev = P_posterior;
    }

    return result;
}
