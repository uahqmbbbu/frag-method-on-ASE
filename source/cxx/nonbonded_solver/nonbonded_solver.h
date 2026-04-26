#pragma once

#include <string>
#include <tuple>
#include <vector>

/// NoCutoff nonbonded solver — LJ + Coulomb for all non-excluded pairs.
class NonBondedSolver {
  public:
    /// Deserialise OpenMM system XML and extract force field parameters.
    explicit NonBondedSolver(
        const std::string &system_xml,
        const std::vector<std::pair<int, int>> &exclude_pair);

    /// Compute energy (kJ/mol) and forces (kJ/mol/nm) for positions (nm).
    std::tuple<double, std::vector<double>>
    compute(const std::vector<double> &positions) const;

    int num_atoms() const { return N; }

  private:
    std::size_t N = 0;

    void get_nonbonded_params(const std::string &system_xml);
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
