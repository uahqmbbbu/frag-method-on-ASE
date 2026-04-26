#pragma once

#include "meika/system.h"

#include "mace_solver/mace_solver.h"
#include "nonbonded_solver/nonbonded_solver.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;

struct GhostH {
    int root;              // global atom index the H attaches to
    std::vector<int> cons; // atoms defining the direction
    double bond_length;
};

struct FragInfo {
    std::vector<int> atoms;     // global atom indices of real atoms
    std::vector<GhostH> ghosts; // ghost H caps
    int sign;                   // +1 for cap-body, -1 for concap
};

class EEGMFCCSolver {
  public:
    EEGMFCCSolver(const std::string &pdb_file, const std::string &model_path,
                  const std::string &precision = "fp32",
                  const std::string &device = "cpu", int batch_size = 128,
                  const std::string &system_xml = "");

    std::tuple<double, py::array_t<double>>
    compute_qm(py::array_t<double> positions_py);

    std::tuple<double, py::array_t<double>>
    compute_mm(py::array_t<double> positions_py);

  private:
    meika::system::System sys_;
    size_t N = 0;
    std::vector<int> atomic_numbers_;
    std::vector<FragInfo> fragments_;
    std::vector<std::pair<int, int>> exclude_pairs_;

    std::unique_ptr<MaceSolver> mace_solver_;
    std::unique_ptr<NonBondedSolver> nb_solver_;

    void map_name_to_z();
    std::vector<std::vector<int>> frag2atom();
    std::vector<std::pair<int, int>> split_chain() const;
    void pre_frag();
};
