#pragma once

#include "meika/system.h"

#include "mace_solver/mace_solver.h"
#include "nonbonded_solver/nonbonded_solver.h"

#include <pybind11/numpy.h>

namespace py = pybind11;

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

class EEGMFCCSolver {
  public:
    EEGMFCCSolver(const std::string &pdb_file, const std::string &model_path,
                  const std::string &precision = "fp32",
                  const std::string &device = "cpu", int batch_size = 128,
                  const std::string &system_xml = "");

    std::tuple<double, py::array_t<double>>
    compute(py::array_t<double> positions_py);

  private:
    struct GhostH {
        int root;
        std::vector<int> cons;
        double bond_length;
    };

    struct FragInfo {
        std::vector<int> atoms;
        std::vector<GhostH> ghosts;
        int sign; // +1 cap-body, -1 concap
    };

    meika::system::System sys_;
    size_t N = 0;
    std::vector<int> atomic_numbers_;
    std::vector<FragInfo> fragments_;
    std::vector<std::pair<int, int>> exclude_pairs_;

    std::unique_ptr<MaceSolver> mace_solver_;
    std::unique_ptr<NonBondedSolver> nb_solver_;

    void build_atomic_numbers();
    std::vector<std::vector<int>> frag2atom();
    std::vector<std::pair<int, int>> split_chain() const;
    void build_fragments();
    void build_exclude_pairs();
    std::tuple<std::vector<int32_t>, std::vector<double>>
    pre_model_input(const FragInfo &frag, const double *pos);

    void dump_fragments(const std::vector<int32_t> &z,
                        const std::vector<double> &coord) const;
};
