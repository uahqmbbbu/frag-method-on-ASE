#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/numpy.h>

namespace OpenMM {
class Context;
class System;
class Integrator;
} // namespace OpenMM

namespace py = pybind11;

/// Evaluate energy & forces via a full OpenMM Context.
class OpenMMSolver {
  public:
    OpenMMSolver(const std::string &system_xml,
                 const std::string &integrator_xml);

    size_t num_atoms() const { return N; }

    std::tuple<double, py::array_t<double>>
    compute(py::array_t<double> positions_py);

  private:
    size_t N = 0;
    std::unique_ptr<OpenMM::System> system_;
    std::unique_ptr<OpenMM::Integrator> integrator_;
    std::unique_ptr<OpenMM::Context> context_;
};
