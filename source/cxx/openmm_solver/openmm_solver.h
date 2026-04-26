#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace OpenMM {
class Context;
class System;
class Integrator;
} // namespace OpenMM

/// Evaluate energy & forces via a full OpenMM Context.
class OpenMMSolver {
  public:
    OpenMMSolver(const std::string &system_xml,
                 const std::string &integrator_xml);

    size_t num_atoms() const { return N; }

    std::tuple<double, std::vector<double>>
    compute(const std::vector<double> &positions);

  private:
    size_t N = 0;
    std::unique_ptr<OpenMM::System> system_;
    std::unique_ptr<OpenMM::Integrator> integrator_;
    std::unique_ptr<OpenMM::Context> context_;
};
