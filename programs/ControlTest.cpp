#include <iostream>
#include "Linear.h"
 

int main() {


    Eigen::MatrixXd A(2, 2);
    A(0, 0) = 0;
    A(1, 0) = 0;
    A(0, 1) = 1;
    A(1, 1) = 0;

    Eigen::MatrixXd B(2, 1); B << 0, 1;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1);

    LinearSystems::StateSpace sys{A, B};
    LinearSystems::CostMatrices cost{Q, R};
    
    auto [K, P] = LinearSystems::LQR(sys, cost);

    std::cout << "K: " << K << std::endl;
    std::cout << "P: " << P << std::endl;
}