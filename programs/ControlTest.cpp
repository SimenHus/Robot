#include <iostream>
#include <fstream>
#include "Linear.h"
#include "Simulation.h"
 

int main() {

    

    const float tau = 200;
    const float q = 25*25;
    const float r = 5;
    const double T = 1.0;
    Eigen::MatrixXd A(4, 4); A << 0, 0, 1, 0,
                                  0, 0, 0, 1,
                                  0, 0, -1/tau, 0,
                                  0, 0, 0, -1/tau;

    std::cout << "Exp test: " << std::endl;
    std::cout << A << std::endl;
    std::cout << A.exp() << std::endl;
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(4, 1);
    Eigen::MatrixXd C(2, 4); C << 1, 0, 0, 0,
                                  0, 1, 0, 0;


    Eigen::MatrixXd G(4, 2); G << 0, 0, 0, 0, -1/tau, 0, 0, -1/tau;
    Eigen::MatrixXd Q = G * (Eigen::MatrixXd::Identity(2, 2) * q) * G.transpose();
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2) * r*r;

    LinearSystems::StateSpace sys(A, B, C);
    LinearSystems::NoiseModel cost(Q, R);
    
    const int iterations = 600;
    // const float pathRadius = 50;
    // Eigen::VectorXd startPos = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd x0(4); x0 << 0, 0, 0, 0;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(4, 4);
    
    LinearSystems::NoiseModel discNoise = cost.getDiscrete(T, sys);
    LinearSystems::StateSpace discSys = sys.getDiscrete(T);
    LinearSystems::KalmanFilter filter(discSys, discNoise, x0, P0);


    // std::vector<Eigen::VectorXd> groundTruth = Simulation::circularPath(iterations, pathRadius, startPos);
    std::vector<Eigen::VectorXd> groundTruth = Simulation::circularPath(iterations, 50.0, Eigen::VectorXd::Zero(2));
    
    const float simulatedDisturbance = 5;
    std::cout << "Simulating..." << std::endl;
    std::vector<Eigen::VectorXd> KFResult = Simulation::KalmanFilter(groundTruth, simulatedDisturbance, filter);
    std::cout << "Kalman sim ok" << std::endl;

    std::ofstream outputFile;
    outputFile.open("kalmanOutput.csv");
    for (const auto &states : KFResult) {
        for (const auto &state : states) {
            outputFile << state << ",";
        }
        outputFile << "\n";
    }
    outputFile.close();


    std::ofstream groundTruthFile;
    groundTruthFile.open("groundTruth.csv");
    for (const auto &states : groundTruth) {
        for (const auto &state : states) {
            groundTruthFile << state << ",";
        }
        groundTruthFile << "\n";
    }
    groundTruthFile.close();

    return 0;
}