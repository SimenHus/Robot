#include <iostream>
#include <fstream>
#include "Linear.h"
#include "NonLinear.h"
#include "Simulation.h"
 

int main() {

    const float tau = 1;
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
    Eigen::MatrixXd C(2, 4); C << 1, 0, 0, 0,
                                  0, 1, 0, 0;


    Eigen::MatrixXd G(4, 2); G << 0, 0, 0, 0, -1/tau, 0, 0, -1/tau;
    Eigen::MatrixXd Q =Eigen::MatrixXd::Identity(2, 2) * q;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2) * r*r;

    LinearSystems::StateSpace sys;
    sys.A = A; sys.C = C; sys.G = G; sys.Q = Q; sys.R = R;
    
    const int iterations = 100;
    const float pathRadius = 100;
    Eigen::VectorXd x0(4); x0 << 0, 0, 0, 0;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(4, 4);
    
    LinearSystems::StateSpace discSys = sys.getDiscrete(T);

    // std::vector<Eigen::VectorXd> groundTruth = Simulation::circularPath(iterations, pathRadius, startPos);
    std::vector<Eigen::VectorXd> groundTruth = Simulation::circularPath(iterations, pathRadius, Eigen::VectorXd::Zero(2));
    
    const float simulatedDisturbance = 3;
    std::cout << "Simulating..." << std::endl;
    std::vector<Eigen::VectorXd> KFResult = Simulation::KalmanFilter(groundTruth, simulatedDisturbance, discSys, x0, P0);
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