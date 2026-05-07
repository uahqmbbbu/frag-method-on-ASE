#include "meika/unit.h"

#include "nonbonded_solver.h"

#include <algorithm>
#include <cmath>
#include <cstring>

NonBondedSolver::NonBondedSolver(
    const std::vector<double> &charges,
    const std::vector<double> &sigma,   // nm
    const std::vector<double> &epsilon, // kJ/mol
    const std::vector<std::vector<double>> &exceptions,
    const std::vector<std::pair<int, int>> &exclude_pairs)
    : exclude_pairs_(exclude_pairs) {

    N = charges.size();
    charges_ = charges;
    sigma_.resize(N);
    epsilon_.resize(N);
#pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        sigma_[i] = sigma[i] * meika::unit::NANOMETER2ANGSTROM;
        epsilon_[i] = epsilon[i] / meika::unit::EV2KJ;
    }

    specials_.reserve(exceptions.size());
    for (const auto &exc : exceptions) {
        int i = static_cast<int>(exc[0]);
        int j = static_cast<int>(exc[1]);
        double qq = exc[2];
        double sig = exc[3];
        double eps = exc[4];
        specials_.push_back({i, j, qq, sig * meika::unit::NANOMETER2ANGSTROM,
                             eps / meika::unit::EV2KJ});
    }

    precalc_matrix();
}

void NonBondedSolver::precalc_matrix() {
    qq_table.resize(N * N);
#pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            const double value = charges_[i] * charges_[j] *
                                 meika::unit::HATREE2EV /
                                 meika::unit::ANGSTROM2BOHR;
            qq_table[i * N + j] = qq_table[j * N + i] = value;
        }
    }

    lj1.resize(N * N);
    lj2.resize(N * N);
    lj3.resize(N * N);
    lj4.resize(N * N);

#pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            const double epsilon_ij = std::sqrt(epsilon_[i] * epsilon_[j]);
            const double sigma_ij = (sigma_[i] + sigma_[j]) / 2;
            const double sigma2 = sigma_ij * sigma_ij;
            const double sigma6 = sigma2 * sigma2 * sigma2;
            const double sigma12 = sigma6 * sigma6;

            const size_t idx1 = i * N + j;
            const size_t idx2 = j * N + i;

            const double lj3_val = 4.0 * epsilon_ij * sigma12;
            const double lj4_val = 4.0 * epsilon_ij * sigma6;
            const double lj1_val = 24.0 * 2.0 * epsilon_ij * sigma12;
            const double lj2_val = 24.0 * epsilon_ij * sigma6;

            lj3[idx1] = lj3[idx2] = lj3_val;
            lj4[idx1] = lj4[idx2] = lj4_val;
            lj1[idx1] = lj1[idx2] = lj1_val;
            lj2[idx1] = lj2[idx2] = lj2_val;
        }
    }

    // 1-2, 1-3, 1-4 modify
#pragma omp parallel for
    for (size_t i = 0; i < specials_.size(); ++i) {
        const auto p1 = specials_[i].p1;
        const auto p2 = specials_[i].p2;

        const auto chargeProd = specials_[i].qq;
        const auto sigma = specials_[i].sigma;
        const auto epsilon = specials_[i].epsilon;

        const size_t idx1 = p1 * N + p2;
        const size_t idx2 = p2 * N + p1;

        qq_table[idx1] = qq_table[idx2] =
            chargeProd * meika::unit::HATREE2EV / meika::unit::ANGSTROM2BOHR;
        const double sigma2 = sigma * sigma;
        const double sigma6 = sigma2 * sigma2 * sigma2;
        const double sigma12 = sigma6 * sigma6;

        lj3[idx1] = lj3[idx2] = 4.0 * epsilon * sigma12;
        lj4[idx1] = lj4[idx2] = 4.0 * epsilon * sigma6;
        lj1[idx1] = lj1[idx2] = 24.0 * 2.0 * epsilon * sigma12;
        lj2[idx1] = lj2[idx2] = 24.0 * epsilon * sigma6;
    }

    // exclude_pair modify
#pragma omp parallel for
    for (size_t i = 0; i < exclude_pairs_.size(); ++i) {
        const auto p1 = exclude_pairs_[i].first;
        const auto p2 = exclude_pairs_[i].second;

        const size_t idx1 = p1 * N + p2;
        const size_t idx2 = p2 * N + p1;

        qq_table[idx1] = qq_table[idx2] = 0.0;
        lj3[idx1] = lj3[idx2] = 0.0;
        lj4[idx1] = lj4[idx2] = 0.0;
        lj1[idx1] = lj1[idx2] = 0.0;
        lj2[idx1] = lj2[idx2] = 0.0;
    }
}

double NonBondedSolver::compute(const double *positions, double *forces) const {
    double energy = 0.0;

#pragma omp parallel for reduction(+ : energy) schedule(static)
    for (size_t i = 0; i < N; ++i) {
        const double xi = positions[3 * i + 0];
        const double yi = positions[3 * i + 1];
        const double zi = positions[3 * i + 2];
        const size_t base = i * N;

        for (size_t j = i + 1; j < N; ++j) {
            const double dx = xi - positions[3 * j + 0];
            const double dy = yi - positions[3 * j + 1];
            const double dz = zi - positions[3 * j + 2];
            const double rsq = dx * dx + dy * dy + dz * dz;
            const double rinv = 1.0 / std::sqrt(rsq);
            const double r2inv = rinv * rinv;
            const double r6inv = r2inv * r2inv * r2inv;

            const size_t idx = base + j;

            const double e_vdwl = r6inv * (lj3[idx] * r6inv - lj4[idx]);
            const double f_vdwl = r6inv * (lj1[idx] * r6inv - lj2[idx]);

            const double e_coul = qq_table[idx] * rinv;
            const double f_coul = e_coul;

            const double epair = e_vdwl + e_coul;
            const double fpair = (f_coul + f_vdwl) * r2inv;

            energy += epair;

#pragma omp atomic
            forces[3 * i + 0] += fpair * dx;
#pragma omp atomic
            forces[3 * i + 1] += fpair * dy;
#pragma omp atomic
            forces[3 * i + 2] += fpair * dz;
#pragma omp atomic
            forces[3 * j + 0] -= fpair * dx;
#pragma omp atomic
            forces[3 * j + 1] -= fpair * dy;
#pragma omp atomic
            forces[3 * j + 2] -= fpair * dz;
        }
    }

    return energy;
}
