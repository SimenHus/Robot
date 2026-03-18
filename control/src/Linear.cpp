#include "Linear.h"

using namespace LinearSystems;

StateSpace::StateSpace(Eigen::MatrixXd A_, Eigen::MatrixXd B_)
    : A(A_), B(B_) {C = Eigen::MatrixXd::Zero(A.rows(), A.cols()); D = Eigen::MatrixXd::Zero(C.rows(), B.cols());}
StateSpace::StateSpace(Eigen::MatrixXd A_, Eigen::MatrixXd B_, Eigen::MatrixXd C_) 
    : A(A_), B(B_), C(C_) {D = Eigen::MatrixXd::Zero(C.rows(), B.cols());}
StateSpace::StateSpace(Eigen::MatrixXd A_, Eigen::MatrixXd B_, Eigen::MatrixXd C_, Eigen::MatrixXd D_) 
    : A(A_), B(B_), C(C_), D(D_) {}


CostMatrices::CostMatrices(Eigen::MatrixXd Q_, Eigen::MatrixXd R_)
    : Q(Q_), R(R_) {N = Eigen::MatrixXd::Zero(Q.rows(), R.cols());}
CostMatrices::CostMatrices(Eigen::MatrixXd Q_, Eigen::MatrixXd R_, Eigen::MatrixXd N_)
    : Q(Q_), R(R_), N(N_) {}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LinearSystems::CARE(const StateSpace &sys, const CostMatrices &cost,
        const float tolerance, const uint iterations, float dt) {

    Eigen::MatrixXd P = cost.Q;
    Eigen::MatrixXd K;

    Eigen::MatrixXd R_inv = cost.R.inverse();
    Eigen::MatrixXd AT = sys.A.transpose();
    Eigen::MatrixXd BT = sys.B.transpose();
    Eigen::MatrixXd NT = cost.N.transpose();
    Eigen::MatrixXd BR_invBT = sys.B*R_inv*BT;
    
    Eigen::MatrixXd P_dot;
    for (uint i = 0; i < iterations; i++) {
        P_dot = AT*P + P*sys.A - (P*sys.B + cost.N)*R_inv*(BT*P + NT) + cost.Q;
        P = P + P_dot*dt;

        if (std::abs(P_dot.norm()) <= tolerance) {
            K = R_inv*(BT*P + NT);
            break;
        }
    }

    return {K, P};
}


std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LinearSystems::DARE(const StateSpace &sys, const CostMatrices &cost,
        const float tolerance, const uint iterations) {

    Eigen::MatrixXd P = cost.Q;
    Eigen::MatrixXd F;

    Eigen::MatrixXd R_inv = cost.R.inverse();
    Eigen::MatrixXd AT = sys.A.transpose();
    Eigen::MatrixXd BT = sys.B.transpose();
    Eigen::MatrixXd NT = cost.N.transpose();
    Eigen::MatrixXd BR_invBT = sys.B*R_inv*BT;
    
    Eigen::MatrixXd P_dot;
    for (uint i = 0; i < iterations; i++) {
        P_dot = (AT*P*sys.B + cost.N)*(cost.R + BT*P*sys.B).inverse()*(BT*P*sys.A + NT) + cost.Q;
        P = P - P_dot;

        if (fabs(P_dot.norm()) <= tolerance) {
            F = (cost.R + BT*P*sys.B).inverse() * (BT*P*sys.A + NT);
            break;
        }
    }

    return {F, P};
}




LQR::LQR(const StateSpace &sys, const CostMatrices &cost)
    : _sys(sys), _cost(cost) {
    auto [K, P] = CARE(_sys, _cost);
    _K = K; _P = P;
    _Kr = (sys.C*(sys.B*_K - sys.A).inverse()*sys.B).inverse();
}