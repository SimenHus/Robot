#pragma once
#include <iostream>
#include <Eigen/Dense>
#include <optional>


namespace LinearSystems {


    struct StateSpace {
        Eigen::MatrixXd A;
        Eigen::MatrixXd B;
        std::optional<Eigen::MatrixXd> C;
        std::optional<Eigen::MatrixXd> D;
    };

    struct CostMatrices {
        Eigen::MatrixXd Q;
        Eigen::MatrixXd R;
        std::optional<Eigen::MatrixXd> N;
    };


    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LQR(const StateSpace &sys, const CostMatrices &cost,
        const float tolerance = 1e-5, const uint iterations = 1e5, float dt = 1e-3);

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LQRDiscrete(const StateSpace &sys, const CostMatrices &cost,
        const float tolerance = 1e-5, const uint iterations = 1e5);

    
};