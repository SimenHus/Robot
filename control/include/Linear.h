#pragma once
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
#include <iostream>
#include <optional>
#include <random>

namespace LinearSystems {

Eigen::VectorXd sampleGaussian(const Eigen::MatrixXd &cov, std::mt19937 &gen);
double gaussianLogLikelihood(const Eigen::VectorXd &x, const Eigen::VectorXd &mean, const Eigen::MatrixXd &cov);

struct StateSpace {
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    Eigen::MatrixXd C;
    Eigen::MatrixXd D;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd G;

    Eigen::VectorXd process(const Eigen::VectorXd &x, const std::optional<Eigen::VectorXd> &u = std::nullopt) const;
    Eigen::VectorXd measure(const Eigen::VectorXd &x, const std::optional<Eigen::VectorXd> &u = std::nullopt) const;
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




namespace KalmanFilter {
    // Standalone functions used in kalman filtering
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> predict(const StateSpace &sys, const Eigen::VectorXd &x, const Eigen::MatrixXd &P, const std::optional<Eigen::VectorXd> &u = std::nullopt);
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> update(const StateSpace &sys, const Eigen::VectorXd &x, const Eigen::MatrixXd &P, const Eigen::VectorXd &z);
    Eigen::MatrixXd innovationCovariance(const StateSpace &sys, const Eigen::MatrixXd &P);

    class Filter { // If kalman filter is needed as object
    private:
        Eigen::VectorXd _x;
        Eigen::MatrixXd _P;
        const StateSpace &_sys;
    public:
        inline void predict(const std::optional<Eigen::VectorXd> &u = std::nullopt) {auto [x, P] = KalmanFilter::predict(_sys, _x, _P, u); _x = x; _P = P;}
        inline void update(const Eigen::VectorXd &z) {auto [x, P] = KalmanFilter::update(_sys, _x, _P, z); _x = x; _P = P;}
        inline Eigen::VectorXd getCurrentState() {return _x;}
        inline Eigen::MatrixXd getCurrentCovariance() {return _P;}
        inline Eigen::MatrixXd innovationCovariance() {return KalmanFilter::innovationCovariance(_sys, _P);}
        inline Eigen::VectorXd measure() const {return _sys.measure(_x);}
    };

}; // Kalman Filter end


    
}; // Namespace end