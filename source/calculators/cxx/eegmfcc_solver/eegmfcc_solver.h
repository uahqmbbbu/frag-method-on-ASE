#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "meika/system.h"
#include "mace_solver/mace_solver.h"

namespace py = pybind11;

// Ghost hydrogen cap
struct GhostH {
    int root;              // global atom index the H attaches to
    std::vector<int> cons; // atoms defining the direction
    double bond_length;
};

// Pre-stored fragment specification
struct FragInfo {
    std::vector<int> atoms;     // global atom indices of real atoms
    std::vector<GhostH> ghosts; // ghost H caps
    int sign;                   // +1 for cap-body, -1 for concap
};

class EEGMFCCSolver {
  public:
    EEGMFCCSolver(const std::string &pdb_file,
                  const std::string &model_path,
                  const std::string &precision = "fp32",
                  const std::string &device = "cpu",
                  int batch_size = 64);

    std::tuple<double, py::array_t<double>>
    compute(py::array_t<int32_t> atomic_numbers_py,
            py::array_t<double> positions_py);

  private:
    meika::system::System sys_;
    int n_atoms_ = 0;
    std::vector<int> atomic_numbers_;
    std::vector<FragInfo> fragments_;
    std::unique_ptr<MaceSolver> solver_;

    void map_name_to_z();
    std::vector<std::vector<int>> frag2atom();
    std::vector<std::pair<int, int>> split_chain() const;
    void pre_frag();
};
