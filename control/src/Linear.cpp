#include "Linear.h"

using namespace LinearSystems;


Eigen::VectorXd StateSpace::process(const Eigen::VectorXd &x) const {
    if (A.size() == 0) {return x;}
    else {return A*x;}
}

Eigen::VectorXd StateSpace::process(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {
    Eigen::MatrixXd Ax = process(x);
    if (B.size() == 0) {return Ax;}
    else {return Ax + B*u;}
}

Eigen::VectorXd StateSpace::measure(const Eigen::VectorXd &x) const {
    if (C.size() == 0) {return x;}
    else {return C*x;}
}

Eigen::VectorXd StateSpace::measure(const Eigen::VectorXd &x, const Eigen::VectorXd &u) const {
    Eigen::MatrixXd Cx = measure(x);
    if (D.size() == 0) {return Cx;}
    else {return Cx + D*u;}
}


StateSpace StateSpace::getDiscrete(const double sampleTime) const {
    // https://en.wikipedia.org/wiki/Discretization
    uint n = A.rows();
    StateSpace result;
    if (C.size() != 0) {result.C = C;}
    if (D.size() != 0) {result.D = D;}
    if (R.size() != 0) {result.R = R/sampleTime;}
    if (A.size() != 0) {result.A = (A*sampleTime).eval().exp();}
    else {return result;}
    // All states below are dependant on A

    if (B.size() != 0) {
        uint m = B.rows();
        Eigen::MatrixXd ABBlock = Eigen::MatrixXd::Zero(n + m, n + m);
        ABBlock.topLeftCorner(n, n) = A;
        ABBlock.topRightCorner(n, m) = B;
        Eigen::MatrixXd expResult = (ABBlock*sampleTime).eval().exp();
        result.B = expResult.block(0, n, n, m);
    }

    if (Q.size() != 0) {
        Eigen::MatrixXd AQBlock = Eigen::MatrixXd::Zero(2*n, 2*n);
        AQBlock.topLeftCorner(n, n) = A;
        AQBlock.bottomRightCorner(n, n) = -A.transpose().eval();
        if (G.size() != 0) {
            AQBlock.topRightCorner(n, n) = G*Q*G.transpose().eval();
        }
        else {
            AQBlock.topRightCorner(n, n) = Q;
        }
        Eigen::MatrixXd expResult = (AQBlock*sampleTime).eval().exp();
        result.Q = expResult.block(0, n, n, n)*expResult.block(0, 0, n, n).transpose().eval();
    }

    return result;
}



std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LinearSystems::CARE(const StateSpace &sys,
        const float tolerance, const uint iterations, float dt) {

    Eigen::MatrixXd P = sys.Q;
    Eigen::MatrixXd K;

    Eigen::MatrixXd R_inv = sys.R.inverse();
    Eigen::MatrixXd AT = sys.A.transpose();
    Eigen::MatrixXd BT = sys.B.transpose();
    Eigen::MatrixXd BR_invBT = sys.B*R_inv*BT;
    
    Eigen::MatrixXd P_dot;
    for (uint i = 0; i < iterations; i++) {
        P_dot = AT*P + P*sys.A - P*sys.B*R_inv*BT*P + sys.Q;
        P = P + P_dot*dt;

        if (std::abs(P_dot.norm()) <= tolerance) {
            K = R_inv*BT*P;
            break;
        }
    }

    return {K, P};
}


std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LinearSystems::DARE(const StateSpace &sys,
        const float tolerance, const uint iterations) {

    Eigen::MatrixXd P = sys.Q;
    Eigen::MatrixXd F;

    Eigen::MatrixXd R_inv = sys.R.inverse();
    Eigen::MatrixXd AT = sys.A.transpose();
    Eigen::MatrixXd BT = sys.B.transpose();
    Eigen::MatrixXd BR_invBT = sys.B*R_inv*BT;
    
    Eigen::MatrixXd P_prev = P;
    for (uint i = 0; i < iterations; i++) {
        P = AT*P_prev*sys.A - AT*P_prev*sys.B*(sys.R + BT*P_prev*sys.B).inverse().eval()*BT*P_prev*sys.A + sys.Q;

        if (std::abs(P_prev.norm() - P.norm()) <= tolerance) {
            F = (sys.R + BT*P*sys.B).inverse().eval() * BT*P*sys.A;
            break;
        }
        P_prev = P;
    }

    return {F, P};
}




LQR::LQR(const StateSpace &sys)
    : _sys(sys) {
    auto [K, P] = DARE(_sys);
    _K = K; _P = P;
    _Kr = (sys.C*(sys.B*_K - sys.A).inverse().eval()*sys.B).inverse().eval();
}

Eigen::VectorXd KalmanFilter::predict() {
    _x = _sys.process(_x);
    _P = _sys.A*_P*_sys.A.transpose().eval() + _sys.Q;
    return _x;
}
Eigen::VectorXd KalmanFilter::predict(const Eigen::VectorXd &u) {
    _x = _sys.process(_x), u;
    _P = _sys.A*_P*_sys.A.transpose().eval() + _sys.Q;
    return _x;
}


Eigen::VectorXd KalmanFilter::update(const Eigen::VectorXd &z) {
    Eigen::MatrixXd CT = _sys.C.transpose().eval();
    Eigen::VectorXd y = z - _sys.C*_x;
    Eigen::MatrixXd S = _sys.C*_P*CT + _sys.R;
    Eigen::MatrixXd K = _P * CT * S.ldlt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(_P.rows(), _P.cols());

    _x = _x + K*y;
    _P = (I-K*_sys.C)*_P*(I-K*_sys.C).transpose().eval() + K*_sys.R*K.transpose().eval();
    return _x;
}