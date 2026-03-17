#include "Linear.h"

using namespace LinearSystems;

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LinearSystems::LQR(const StateSpace &sys, const CostMatrices &cost,
        const float tolerance, const uint iterations, float dt) {

    Eigen::MatrixXd P = cost.Q;
    Eigen::MatrixXd K;

    Eigen::MatrixXd R_inv = cost.R.inverse();
    Eigen::MatrixXd AT = sys.A.transpose();
    Eigen::MatrixXd BT = sys.B.transpose();
    Eigen::MatrixXd BR_invBT = sys.B*R_inv*BT;
    
    Eigen::MatrixXd P_dot;
    for (uint i = 0; i < iterations; i++) {
        P_dot = AT*P + P*sys.A - P*BR_invBT*P + cost.Q;
        P = P + P_dot*dt;

        if (fabs(P_dot.norm()) <= tolerance) {
            K = R_inv*BT*P;
            break;
        }
    }

    return {K, P};
}


std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LinearSystems::LQRDiscrete(const StateSpace &sys, const CostMatrices &cost,
        const float tolerance, const uint iterations) {

    Eigen::MatrixXd P = cost.Q;
    Eigen::MatrixXd F;

    Eigen::MatrixXd R_inv = cost.R.inverse();
    Eigen::MatrixXd AT = sys.A.transpose();
    Eigen::MatrixXd BT = sys.B.transpose();
    Eigen::MatrixXd BR_invBT = sys.B*R_inv*BT;
    
    Eigen::MatrixXd P_dot;
    for (uint i = 0; i < iterations; i++) {
        P_dot = AT*P*sys.B*(cost.R + BT*P*sys.B).inverse()*BT*P*sys.A + cost.Q;
        P = P - P_dot;

        if (fabs(P_dot.norm()) <= tolerance) {
            F = (cost.R + BT*P*sys.B).inverse() * BT*P*sys.A;
            break;
        }
    }

    return {F, P};
}
