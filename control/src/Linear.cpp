#include "Linear.h"

using namespace LinearSystems;

StateSpace::StateSpace(Eigen::MatrixXd A_, Eigen::MatrixXd B_)
    : A(A_), B(B_) {C = Eigen::MatrixXd::Zero(A.rows(), A.cols()); D = Eigen::MatrixXd::Zero(C.rows(), B.cols());}
StateSpace::StateSpace(Eigen::MatrixXd A_, Eigen::MatrixXd B_, Eigen::MatrixXd C_) 
    : A(A_), B(B_), C(C_) {D = Eigen::MatrixXd::Zero(C.rows(), B.cols());}
StateSpace::StateSpace(Eigen::MatrixXd A_, Eigen::MatrixXd B_, Eigen::MatrixXd C_, Eigen::MatrixXd D_) 
    : A(A_), B(B_), C(C_), D(D_) {}


StateSpace StateSpace::getDiscrete(const double sampleTime) const {
    // https://en.wikipedia.org/wiki/Discretization
    uint n = A.rows();
    uint m = B.cols();
    Eigen::MatrixXd block = Eigen::MatrixXd::Zero(n + m, n + m);
    block.topLeftCorner(n, n) = A;
    block.topRightCorner(n, m) = B;
    Eigen::MatrixXd expResult = (block*sampleTime).eval().exp();
    Eigen::MatrixXd discA = expResult.block(0, 0, n, n);
    Eigen::MatrixXd discB = expResult.block(0, n, n, m);
    return StateSpace(discA, discB, C, D);
}


NoiseModel NoiseModel::getDiscrete(const double sampleTime, const StateSpace &sys) const {
    uint n = sys.A.rows();
    Eigen::MatrixXd block = Eigen::MatrixXd::Zero(2*n, 2*n);
    block.topLeftCorner(n, n) = sys.A;
    block.topRightCorner(n, n) = Q;
    block.bottomRightCorner(n, n) = -sys.A.transpose().eval();

    Eigen::MatrixXd expResult = (block*sampleTime).eval().exp();
    Eigen::MatrixXd discR = R/sampleTime;
    Eigen::MatrixXd discQ = expResult.block(0, n, n, n)*expResult.block(0, 0, n, n).transpose().eval();
    return NoiseModel(discQ, discR);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LinearSystems::CARE(const StateSpace &sys, const NoiseModel &cost,
        const float tolerance, const uint iterations, float dt) {

    Eigen::MatrixXd P = cost.Q;
    Eigen::MatrixXd K;

    Eigen::MatrixXd R_inv = cost.R.inverse();
    Eigen::MatrixXd AT = sys.A.transpose();
    Eigen::MatrixXd BT = sys.B.transpose();
    Eigen::MatrixXd BR_invBT = sys.B*R_inv*BT;
    
    Eigen::MatrixXd P_dot;
    for (uint i = 0; i < iterations; i++) {
        P_dot = AT*P + P*sys.A - P*sys.B*R_inv*BT*P + cost.Q;
        P = P + P_dot*dt;

        if (std::abs(P_dot.norm()) <= tolerance) {
            K = R_inv*BT*P;
            break;
        }
    }

    return {K, P};
}


std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LinearSystems::DARE(const StateSpace &sys, const NoiseModel &cost,
        const float tolerance, const uint iterations) {

    Eigen::MatrixXd P = cost.Q;
    Eigen::MatrixXd F;

    Eigen::MatrixXd R_inv = cost.R.inverse();
    Eigen::MatrixXd AT = sys.A.transpose();
    Eigen::MatrixXd BT = sys.B.transpose();
    Eigen::MatrixXd BR_invBT = sys.B*R_inv*BT;
    
    Eigen::MatrixXd P_prev = P;
    for (uint i = 0; i < iterations; i++) {
        P = AT*P_prev*sys.A - AT*P_prev*sys.B*(cost.R + BT*P_prev*sys.B).inverse().eval()*BT*P_prev*sys.A + cost.Q;

        if (std::abs(P_prev.norm() - P.norm()) <= tolerance) {
            F = (cost.R + BT*P*sys.B).inverse().eval() * BT*P*sys.A;
            break;
        }
        P_prev = P;
    }

    return {F, P};
}




LQR::LQR(const StateSpace &sys, const NoiseModel &cost)
    : _sys(sys), _cost(cost) {
    auto [K, P] = DARE(_sys, _cost);
    _K = K; _P = P;
    _Kr = (sys.C*(sys.B*_K - sys.A).inverse().eval()*sys.B).inverse().eval();
}

Eigen::VectorXd KalmanFilter::predict() {
    Eigen::VectorXd u = Eigen::VectorXd::Zero(_sys.B.cols());
    return predict(u);
}
Eigen::VectorXd KalmanFilter::predict(const Eigen::VectorXd &u) {
    _x = _sys.A*_x + _sys.B*u;
    _P = _sys.A*_P*_sys.A.transpose().eval() + _noise.Q;
    return _x;
}


Eigen::VectorXd KalmanFilter::update(const Eigen::VectorXd &z) {
    Eigen::MatrixXd CT = _sys.C.transpose().eval();
    Eigen::VectorXd y = z - _sys.C*_x;
    Eigen::MatrixXd S = _sys.C*_P*CT + _noise.R;
    Eigen::MatrixXd K = _P * CT * S.ldlt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(_P.rows(), _P.cols());

    _x = _x + K*y;
    _P = (I-K*_sys.C)*_P*(I-K*_sys.C).transpose().eval() + K*_noise.R*K.transpose().eval();
    return _x;
}