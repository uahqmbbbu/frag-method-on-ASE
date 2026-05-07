#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

/// NoCutoff nonbonded solver — LJ + Coulomb for all non-excluded pairs.
class NonBondedSolver {
  public:
    /// 直接从参数构造（参数由 Python 层从 system.xml 提取）
    NonBondedSolver(
        const std::vector<double> &charges,
        const std::vector<double> &sigma,       // nm
        const std::vector<double> &epsilon,      // kJ/mol
        const std::vector<std::vector<double>>
            &exceptions,                         // [i,j,qq,sig_nm,eps_kjmol]
        const std::vector<std::pair<int, int>> &exclude_pairs);

    size_t num_atoms() const { return N; }

    /// pos: flat n×3 input (Å).  force_out: flat n×3 output, *accumulated* (not
    /// zeroed).
    double compute(const double *pos, double *force_out) const;

  private:
    std::size_t N = 0;

    std::vector<double> charges_;
    std::vector<double> sigma_;
    std::vector<double> epsilon_;

    struct Special {
        int p1, p2;
        double qq;
        double sigma;
        double epsilon;
    };
    std::vector<Special> specials_;

    void precalc_matrix();
    std::vector<double> qq_table;
    std::vector<double> lj1;
    std::vector<double> lj2;
    std::vector<double> lj3;
    std::vector<double> lj4;

    std::vector<std::pair<int, int>> exclude_pairs_;
};
