#include "meika/unit.h"

#include "nonbonded_solver.h"

#include <openmm/NonbondedForce.h>
#include <openmm/serialization/XmlSerializer.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

NonBondedSolver::NonBondedSolver(
    const std::string &system_xml,
    const std::vector<std::pair<int, int>> &exclude_pair)
    : exclude_pairs_(exclude_pair) {
    get_nonbonded_params(system_xml);
    precalc_matrix();
}

void NonBondedSolver::get_nonbonded_params(const std::string &system_xml) {
    std::ifstream xml_file(system_xml);
    if (!xml_file.is_open()) {
        throw std::runtime_error("Cannot open system XML: " + system_xml);
    }
    auto sys_ptr = OpenMM::XmlSerializer::deserialize<OpenMM::System>(xml_file);
    auto &sys = *sys_ptr;

    // ---- find NonbondedForce ----
    const OpenMM::NonbondedForce *nb = nullptr;
    for (int k = 0; k < sys.getNumForces(); ++k) {
        auto &force = sys.getForce(k);
        if (force.getName() == "NonbondedForce") {
            nb = dynamic_cast<const OpenMM::NonbondedForce *>(&force);
            break;
        }
    }
    if (!nb)
        throw std::runtime_error("NonbondedForce not found in system XML.");

    N = nb->getNumParticles();
    charges_.resize(N);
    sigma_.resize(N);
    epsilon_.resize(N);
    for (size_t i = 0; i < N; ++i) {
        double charge, sigma, epsilon;
        nb->getParticleParameters(i, charge, sigma, epsilon);
        charges_[i] = charge;
        sigma_[i] = sigma * meika::unit::NANOMETER2ANGSTROM;
        epsilon_[i] = epsilon / meika::unit::EV2KJ;
    }

    int nex = nb->getNumExceptions();
    specials_.reserve(nex);
    for (int k = 0; k < nex; ++k) {
        int i, j;
        double qq, sig, eps;
        nb->getExceptionParameters(k, i, j, qq, sig, eps);
        if (i > j) std::swap(i, j);

        specials_.push_back({i, j, qq, sig * meika::unit::NANOMETER2ANGSTROM,
                             eps / meika::unit::EV2KJ});
    }
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

    // 为1-2, 1-3, 1-4作用修改矩阵
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

    // 为exclude_pair修改矩阵
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

std::tuple<double, std::vector<double>>
NonBondedSolver::compute(const std::vector<double> &positions) const {

    std::vector<double> forces(N * 3, 0.0);
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

    return {energy, forces};
}
