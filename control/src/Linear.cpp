#include "Linear.h"

using namespace LinearSystems;


Eigen::VectorXd LinearSystems::sampleGaussian(const Eigen::MatrixXd &cov, std::mt19937 &gen) {
    static std::normal_distribution<double> dist(0.0, 1.0);

    Eigen::VectorXd z(cov.rows());
    for (int i = 0; i < z.size(); ++i)
        z(i) = dist(gen);

    Eigen::MatrixXd L = cov.llt().matrixL(); // Cholesky

    return L * z;
}

double LinearSystems::gaussianLogLikelihood(const Eigen::VectorXd &x, const Eigen::VectorXd &mean, const Eigen::MatrixXd &cov) {
    int d = x.size();
    
    Eigen::LDLT<Eigen::MatrixXd> ldlt(cov); // Use cholesky decomposition
    Eigen::MatrixXd cov_inv = ldlt.solve(Eigen::MatrixXd::Identity(d, d));
    double cov_det = ldlt.vectorD().prod();

    Eigen::VectorXd diff = x - mean;
    double mahalanobis_squared = diff.transpose().eval() * cov_inv * diff;

    double log_likelihood = -d / 2.0 * std::log(2 * M_PI) - 
                            0.5 * std::log(cov_det) - 
                            0.5 * mahalanobis_squared;

    return log_likelihood;
}


Eigen::VectorXd StateSpace::process(const Eigen::VectorXd &x, const std::optional<Eigen::VectorXd> &u) const {
    Eigen::MatrixXd Ax = A*x;
    if (u) {return Ax + B*u.value();}
    else {return Ax;}
}

Eigen::VectorXd StateSpace::measure(const Eigen::VectorXd &x, const std::optional<Eigen::VectorXd> &u) const {
    Eigen::MatrixXd Cx = C*x;
    if (u) {return Cx + D*u.value();}
    else {return Cx;}
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
    for (uint i = 0; i < iterations; ++i) {
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
    for (uint i = 0; i < iterations; ++i) {
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


std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::predict(const StateSpace &sys, const Eigen::VectorXd &x, const Eigen::MatrixXd &P, const std::optional<Eigen::VectorXd> &u) {
    Eigen::VectorXd x_priori = sys.process(x, u);
    Eigen::MatrixXd P_priori = sys.A*P*sys.A.transpose().eval() + sys.Q;
    return {x_priori, P_priori};
}

Eigen::MatrixXd KalmanFilter::innovationCovariance(const StateSpace &sys, const Eigen::MatrixXd &P) {
    Eigen::MatrixXd CT = sys.C.transpose().eval();
    Eigen::MatrixXd S = sys.C*P*CT + sys.R;
    return S;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::update(const StateSpace &sys, const Eigen::VectorXd &x, const Eigen::MatrixXd &P, const Eigen::VectorXd &z) {
    Eigen::MatrixXd CT = sys.C.transpose().eval();
    Eigen::VectorXd y = z - sys.C*x;
    Eigen::MatrixXd S = innovationCovariance(sys, P);
    Eigen::MatrixXd K = P * CT * S.ldlt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P.rows(), P.cols());

    Eigen::VectorXd x_posteriori = x + K*y;
    Eigen::MatrixXd P_posteriori = (I-K*sys.C)*P*(I-K*sys.C).transpose().eval() + K*sys.R*K.transpose().eval();
    return {x_posteriori, P_posteriori};
}