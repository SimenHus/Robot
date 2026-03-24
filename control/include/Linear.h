#pragma once
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
#include <iostream>
#include <optional>

namespace LinearSystems {

struct StateSpace {
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    Eigen::MatrixXd C;
    Eigen::MatrixXd D;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd G;

    Eigen::VectorXd process(const Eigen::VectorXd &x) const;
    Eigen::VectorXd measure(const Eigen::VectorXd &x) const;
    Eigen::VectorXd process(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const;
    Eigen::VectorXd measure(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const;
    // inline void print() const {std::cout << A << std::endl << B << std::endl << C << std::endl << D << std::endl;}

    StateSpace getDiscrete(const double sampleTime) const;
};

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> CARE(const StateSpace &sys,
    const float tolerance = 1e-5, const uint iterations = 1e5, float dt = 1e-3);

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> DARE(const StateSpace &sys,
    const float tolerance = 1e-5, const uint iterations = 1e5);


class LQR {
private:
    Eigen::MatrixXd _P;
    Eigen::MatrixXd _K;
    Eigen::MatrixXd _Kr;

    const StateSpace &_sys;

public:
    LQR(const StateSpace &sys);
    inline Eigen::VectorXd calculateGain(const Eigen::VectorXd &x) const {return -_K*x;}
    inline Eigen::VectorXd calculateGain(const Eigen::VectorXd &x, const Eigen::VectorXd &r) const {return _Kr*r - _K*x;}

    inline Eigen::VectorXd process(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {return _sys.process(x, u);}
    inline Eigen::VectorXd measure(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {return _sys.measure(x, u);}
    inline Eigen::VectorXd measure(const Eigen::VectorXd &x) const {return _sys.measure(x);}

};




class KalmanFilter {
private:
    Eigen::MatrixXd _P;
    Eigen::MatrixXd _x;

    const StateSpace &_sys;


public:
    KalmanFilter(const StateSpace &sys, const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0) : _sys(sys), _x(x0), _P(P0) {};
    Eigen::VectorXd predict();
    Eigen::VectorXd predict(const Eigen::VectorXd &u);
    Eigen::VectorXd update(const Eigen::VectorXd &z);

    inline Eigen::VectorXd currentState() const {return _x;}
};





    
}; // Namespace end