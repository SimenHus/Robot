#pragma once
#include <Eigen/Dense>
#include <functional>
#include <optional>
#include <numeric>
#include <vector>
#include <random>

#include "Linear.h"

namespace NonLinearSystems {



struct StateSpace {
    const std::function<Eigen::VectorXd(const Eigen::VectorXd, const std::optional<Eigen::VectorXd>)> processModel;
    const std::function<Eigen::VectorXd(const Eigen::VectorXd, const std::optional<Eigen::VectorXd>)> measurementModel;
    const Eigen::MatrixXd processNoise;
    const Eigen::MatrixXd measurementNoise;
};


    namespace ParticleFilter {


        struct Particle {
            Eigen::VectorXd x;
            double w;
            const StateSpace &sys;
            std::mt19937 &gen;
            std::optional<LinearSystems::KalmanFilter::Filter> kf = std::nullopt;
        };

        std::vector<Particle> initialize(const Particle &particleBase, const uint particleCount);
        void predict(std::vector<Particle> &particles);
        void update(std::vector<Particle> &particles, const Eigen::VectorXd &z);
        void resample(std::vector<Particle> &particles);
        std::vector<int> systematicResample(std::vector<double> &weights, std::mt19937 &gen);


    }; // Particle Filter

}; // Nonlinear systems
