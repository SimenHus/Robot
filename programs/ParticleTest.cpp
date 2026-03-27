#include <iostream>
#include <fstream>
#include <random>
#include <functional>
#include <optional>

#include "Linear.h"
#include "NonLinear.h"
#include "Simulation.h"
 

int main(int argc, char* argv[]) {
    int numParticles = 100;
    if (argc > 1) {
        // Number of particles passed as argument
        numParticles = std::atoi(argv[1]);
        std::cout << "Simulating particle filter with " << numParticles << " particles" << std::endl;
    }

    const float tau = 1;
    const float q = 25*25;
    const float r = 5;
    const double T = 1.0;
    Eigen::MatrixXd A(4, 4); A << 0, 0, 1, 0,
                                  0, 0, 0, 1,
                                  0, 0, -1/tau, 0,
                                  0, 0, 0, -1/tau;

    std::cout << "Exp test: " << std::endl;
    std::cout << A << std::endl;
    std::cout << A.exp() << std::endl;
    Eigen::MatrixXd C(2, 4); C << 1, 0, 0, 0,
                                  0, 1, 0, 0;


    Eigen::MatrixXd G(4, 2); G << 0, 0, 0, 0, -1/tau, 0, 0, -1/tau;
    Eigen::MatrixXd Q =Eigen::MatrixXd::Identity(2, 2) * q;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2) * r*r;

    
    LinearSystems::StateSpace sys;
    sys.A = A; sys.C = C; sys.G = G; sys.Q = Q; sys.R = R;

    
    const int iterations = 100;
    const float pathRadius = 100;
    Eigen::VectorXd x0(4); x0 << 0, 0, 0, 0;
    Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(4, 4);
    
    std::random_device rd;
    std::seed_seq ss{rd(), rd(), rd(), rd()}; 
    std::mt19937 gen(ss);
    
    LinearSystems::StateSpace discSys = sys.getDiscrete(T);
    auto processModel = [&discSys](const Eigen::VectorXd x, const std::optional<Eigen::VectorXd> u) {
        return discSys.process(x, u);
    };
    auto measurementModel = [&discSys](const Eigen::VectorXd x, const std::optional<Eigen::VectorXd> u) {
        return discSys.measure(x, u);
    };
    NonLinearSystems::StateSpace nonlinSys{processModel, measurementModel, discSys.Q, discSys.R};
    NonLinearSystems::ParticleFilter::Particle baseParticle = NonLinearSystems::ParticleFilter::Particle(x0, 0.0, nonlinSys, gen);
    std::vector<NonLinearSystems::ParticleFilter::Particle> particles = NonLinearSystems::ParticleFilter::initialize(baseParticle, numParticles);

    // std::vector<Eigen::VectorXd> groundTruth = Simulation::circularPath(iterations, pathRadius, startPos);
    std::vector<Eigen::VectorXd> groundTruth = Simulation::circularPath(iterations, pathRadius, Eigen::VectorXd::Zero(2));
    

    // SIMULATE KALMAN FILTER
    std::cout << "Simulating..." << std::endl;
    const float simulatedDisturbance = 3;
    std::vector<Eigen::VectorXd> KFResult(groundTruth.size());
    KFResult[0] = x0;

    Eigen::MatrixXd P_prev = P0;
    for (uint i = 1; i < groundTruth.size(); ++i) {
        Eigen::VectorXd measurement = groundTruth[i] + LinearSystems::sampleGaussian(discSys.R, gen);
        auto [x_priori, P_priori] = LinearSystems::KalmanFilter::predict(discSys, KFResult[i-1], P_prev);
        auto [x_posterior, P_posterior] = LinearSystems::KalmanFilter::update(discSys, x_priori, P_priori, measurement);
        KFResult[i] = x_posterior;
        P_prev = P_posterior;
    }
    std::cout << "Kalman sim ok" << std::endl;



    // SIMULATE PARTICLE FILTER
    std::vector<Eigen::VectorXd> PFResult; PFResult.reserve(groundTruth.size());

    for (const auto &gt : groundTruth) {
        Eigen::VectorXd z = gt;
        z += LinearSystems::sampleGaussian(baseParticle.sys.measurementNoise, gen);

        NonLinearSystems::ParticleFilter::predict(particles);
        NonLinearSystems::ParticleFilter::update(particles, z);
        Eigen::VectorXd x_est = NonLinearSystems::ParticleFilter::estimateMean(particles);
        PFResult.push_back(x_est);
    }
    std::cout << "Particle filter sim ok" << std::endl;


    std::ofstream outputFile;
    outputFile.open("kalmanOutput.csv");
    for (const auto &states : KFResult) {
        for (const auto &state : states) {
            outputFile << state << ",";
        }
        outputFile << "\n";
    }
    outputFile.close();

    std::ofstream particleFile;
    particleFile.open("particleOutput.csv");
    for (const auto &states : PFResult) {
        for (const auto &state : states) {
            particleFile << state << ",";
        }
        particleFile << "\n";
    }
    particleFile.close();

    std::ofstream groundTruthFile;
    groundTruthFile.open("groundTruth.csv");
    for (const auto &states : groundTruth) {
        for (const auto &state : states) {
            groundTruthFile << state << ",";
        }
        groundTruthFile << "\n";
    }
    groundTruthFile.close();

    return 0;
}