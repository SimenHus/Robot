#pragma once
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
#include <iostream>

namespace LinearSystems {

struct StateSpace {
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    Eigen::MatrixXd C;
    Eigen::MatrixXd D;

    StateSpace(Eigen::MatrixXd A_, Eigen::MatrixXd B_);
    StateSpace(Eigen::MatrixXd A_, Eigen::MatrixXd B_, Eigen::MatrixXd C_);
    StateSpace(Eigen::MatrixXd A_, Eigen::MatrixXd B_, Eigen::MatrixXd C_, Eigen::MatrixXd D_);

    inline Eigen::VectorXd process(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {return A*x + B*u;}
    inline Eigen::VectorXd observe(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {return C*x + D*u;}
    inline Eigen::VectorXd observe(const Eigen::VectorXd &x) const {return C*x;}
    inline void print() const {std::cout << A << std::endl << B << std::endl << C << std::endl << D << std::endl;}

    StateSpace getDiscrete(const double sampleTime) const;
};

struct NoiseModel {
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;

    NoiseModel(Eigen::MatrixXd Q_, Eigen::MatrixXd R_) : Q(Q_), R(R_) {}
    NoiseModel getDiscrete(const double sampleTime, const StateSpace &sys) const;
    inline void print() const {std::cout << Q << std::endl << R << std::endl;}
};



std::pair<Eigen::MatrixXd, Eigen::MatrixXd> CARE(const StateSpace &sys, const NoiseModel &cost,
    const float tolerance = 1e-5, const uint iterations = 1e5, float dt = 1e-3);

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> DARE(const StateSpace &sys, const NoiseModel &cost,
    const float tolerance = 1e-5, const uint iterations = 1e5);


class LQR {
private:
    Eigen::MatrixXd _P;
    Eigen::MatrixXd _K;
    Eigen::MatrixXd _Kr;

    const StateSpace &_sys;
    const NoiseModel &_cost;

public:
    LQR(const StateSpace &sys, const NoiseModel &cost);
    inline Eigen::VectorXd calculateGain(const Eigen::VectorXd &x) const {return -_K*x;}
    inline Eigen::VectorXd calculateGain(const Eigen::VectorXd &x, const Eigen::VectorXd &r) const {return _Kr*r - _K*x;}

    inline Eigen::VectorXd process(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {return _sys.process(x, u);}
    inline Eigen::VectorXd observe(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {return _sys.observe(x, u);}
    inline Eigen::VectorXd observe(const Eigen::VectorXd &x) const {return _sys.observe(x);}

};




class KalmanFilter {
private:
    Eigen::MatrixXd _P;
    Eigen::MatrixXd _x;

    const StateSpace &_sys;
    const NoiseModel &_noise;


public:
    KalmanFilter(const StateSpace &sys, const NoiseModel &noise, const Eigen::VectorXd &x0, const Eigen::MatrixXd &P0) : _sys(sys), _noise(noise), _x(x0), _P(P0) {};
    Eigen::VectorXd predict();
    Eigen::VectorXd predict(const Eigen::VectorXd &u);
    Eigen::VectorXd update(const Eigen::VectorXd &z);

    inline Eigen::VectorXd currentState() const {return _x;}
};





    
}; // Namespace end