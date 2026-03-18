#include <iostream>
#include <fstream>
#include "Linear.h"
#include "Simulation.h"
 

int main() {

    // Example: https://www.cds.caltech.edu/~murray/courses/cds110/sp2024/L4-3_statefbk-26Apr2024.pdf
    Eigen::MatrixXd A(2, 2); A << 0, 10, -1, 0;
    Eigen::MatrixXd B(2, 1); B << 0, 1;
    Eigen::MatrixXd C(1, 2); C << 1, 1;


    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1);

    LinearSystems::StateSpace sys(A, B, C);
    LinearSystems::CostMatrices cost(Q, R);
    
    LinearSystems::LQR controller(sys, cost);


    Eigen::VectorXd r(1); r << 2;
    const int iterations = 10;
    std::vector<Eigen::VectorXd> reference(iterations);
    Eigen::VectorXd x0(2); x0 << 0, 0;

    std::vector<float> time(iterations);
    for (uint i = 0; i < iterations; i++) {
        time[i] = i;
        reference[i] = r;
    }

    std::vector<Eigen::VectorXd> result = Simulation::LQR(time, controller, x0, reference);

    std::ofstream outputFile;
    outputFile.open("output.csv");
    for (const auto &states : result) {
        for (const auto &state : states) {
            outputFile << state << ",";
        }
        outputFile << "\n";
    }
    outputFile.close();

    return 0;
}