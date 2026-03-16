#include <iostream>
#include "LQR.h"
 

int main() {


    Eigen::MatrixXd A(2, 2);
    A(0, 0) = 3;
    A(1, 0) = 2.5;
    A(0, 1) = -15;
    A(1, 1) = A(1, 0) + A(0, 1);

    Eigen::MatrixXd B(2, 2);
    Eigen::MatrixXd C(2, 2);
    Eigen::MatrixXd D(2, 2);
    
    LQR controller(A, B, C, D);

    std::cout << controller.getMatrixA() << std::endl;
}