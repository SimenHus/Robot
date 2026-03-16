#include "LQR.h"



LQR::LQR(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, const Eigen::MatrixXd &C, const Eigen::MatrixXd &D) 
    : _A(A), _B(B), _C(C), _D(D) {
}



Eigen::MatrixXd LQR::getMatrixA() {
    return _A;
}