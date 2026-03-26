#include "NonLinear.h"


using namespace NonLinearSystems;


std::vector<ParticleFilter::Particle> ParticleFilter::initialize(const ParticleFilter::Particle &baseParticle, const uint particleCount) {
    std::vector<ParticleFilter::Particle> particles(particleCount);
    for (int i = 0; i < particleCount; i++) {
        Eigen::VectorXd x_noised = baseParticle.x + LinearSystems::sampleGaussian(baseParticle.sys.processNoise, baseParticle.gen);
        ParticleFilter::Particle newParticle = baseParticle;
        newParticle.x = x_noised;
        newParticle.w = -std::log(particleCount);
        particles.push_back(newParticle);
    }

    return particles;
}


void ParticleFilter::predict(std::vector<ParticleFilter::Particle> &particles) {
    for (auto &p : particles) {
        Eigen::VectorXd noise = LinearSystems::sampleGaussian(p.sys.processNoise, p.gen); // Noisify current estimate
        p.x = p.sys.processModel(p.x + noise, std::nullopt); // Predict next step with noise added to previous guess
        if (p.kf) {p.kf.value().predict();} // Predict linear states if available
    }
}




std::vector<int> ParticleFilter::systematicResample(const std::vector<double> &weights, std::mt19937 &gen) {
    int N = weights.size();

    // 1. cumweights ← cumsum(weights)
    std::vector<double> cumweights(N);
    std::partial_sum(weights.begin(), weights.end(), cumweights.begin());

    // 2. indicesout ← zeros(N,1)
    std::vector<int> indices_out(N);

    // 3. noise ← rand(1,1)/N
    std::uniform_real_distribution<double> dist(0.0, 1.0 / N);
    double noise = dist(gen);

    // 4. i ← 1  (C++ index = 0)
    int i = 0;

    // 5. for j ← 1:N
    for (int j = 0; j < N; ++j)
    {
        // u_j ← (j-1)/N + noise
        double u = j * (1.0 / N) + noise;

        // while u_j > cumweights[i]
        while (u > cumweights[i])
        {
            i++;
            // safety (shouldn't happen if weights normalized)
            if (i >= N) i = N - 1;
        }

        // indicesout[j] ← i
        indices_out[j] = i;
    }

    return indices_out;
}


void ParticleFilter::resample(std::vector<ParticleFilter::Particle> &particles) {
    const int n = particles.size();
    std::vector<Particle> newParticles(n);
    std::vector<double> weights(n);

    for (auto &p : particles) {
        weights.push_back(p.w);
    }
    std::vector<int> indices = ParticleFilter::systematicResample(weights, particles[0].gen);

    for (int idx : indices)
    {
        newParticles.push_back(particles[idx]);
        newParticles.back().w = - std::log(n); // reset weights
    }

    particles = std::move(newParticles);
}


void ParticleFilter::update(std::vector<ParticleFilter::Particle> &particles, const Eigen::VectorXd &z) {
    double maxLogWeight = std::numeric_limits<double>::infinity(); // Keep track of largest weight
    // Update particle weights and kalman filters
    for (auto &p : particles) {
        Eigen::VectorXd particlePrediction = p.sys.measurementModel(p.x, std::nullopt);
        if (p.kf) {
            Eigen::MatrixXd innovationCovariance = p.kf.value().innovationCovariance();
            p.kf.value().update(z - particlePrediction);
            Eigen::VectorXd linearPrediction = p.kf.value().measure();
            p.w += LinearSystems::gaussianLogLikelihood(z, particlePrediction + linearPrediction, innovationCovariance);
        }
        else {
            p.w += LinearSystems::gaussianLogLikelihood(z, particlePrediction, p.sys.measurementNoise);
        }
        maxLogWeight = std::max(maxLogWeight, p.w); // Check if current weight is highest
    }

    // Normalize particle weights
    double logSum = 0;
    for (auto &p : particles) {
        logSum += std::exp(p.w - maxLogWeight);
    }
    double logSumExp = maxLogWeight + std::log(logSum);
    double sumSquares = 0;
    for (auto &p : particles) {
        p.w -= logSumExp;
        double w = std::exp(p.w);
        sumSquares += w*w;
    }

    double effectiveParticles = 1 / sumSquares;

    // Resample if needed
    if (effectiveParticles < particles.size() / 2) {
        ParticleFilter::resample(particles);
    }
}