#include <iostream>
#include <Eigen/Dense>

class LQR {
private:
    Eigen::MatrixXd _A;
    Eigen::MatrixXd _B;
    Eigen::MatrixXd _C;
    Eigen::MatrixXd _D;


public:

    LQR(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B,
        const Eigen::MatrixXd &C, const Eigen::MatrixXd &D);


    Eigen::MatrixXd getMatrixA();
};