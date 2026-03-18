#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <optional>

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
};

struct CostMatrices {
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd N;

    CostMatrices(Eigen::MatrixXd Q_, Eigen::MatrixXd R_);
    CostMatrices(Eigen::MatrixXd Q_, Eigen::MatrixXd R_, Eigen::MatrixXd N_);
};



std::pair<Eigen::MatrixXd, Eigen::MatrixXd> CARE(const StateSpace &sys, const CostMatrices &cost,
    const float tolerance = 1e-5, const uint iterations = 1e5, float dt = 1e-3);

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> DARE(const StateSpace &sys, const CostMatrices &cost,
    const float tolerance = 1e-5, const uint iterations = 1e5);


class LQR {
private:
    Eigen::MatrixXd _P;
    Eigen::MatrixXd _K;
    Eigen::MatrixXd _Kr;

    const StateSpace &_sys;
    const CostMatrices &_cost;

public:
    LQR(const StateSpace &sys, const CostMatrices &cost);
    inline Eigen::VectorXd calculateGain(const Eigen::VectorXd &x) const {return -_K*x;}
    inline Eigen::VectorXd calculateGain(const Eigen::VectorXd &x, const Eigen::VectorXd &r) const {return _Kr*r - _K*x;}

    inline Eigen::VectorXd process(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {return _sys.process(x, u);}
    inline Eigen::VectorXd observe(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {return _sys.observe(x, u);}
    inline Eigen::VectorXd observe(const Eigen::VectorXd &x) const {return _sys.observe(x);}

};









    
}; // Namespace end