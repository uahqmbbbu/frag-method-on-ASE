#include "meika/unit.h"

#include "openmm_solver.h"

#include <openmm/Context.h>
#include <openmm/Integrator.h>
#include <openmm/Platform.h>
#include <openmm/System.h>
#include <openmm/Vec3.h>
#include <openmm/serialization/XmlSerializer.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

OpenMMSolver::OpenMMSolver(const std::string &system_xml,
                           const std::string &integrator_xml) {
    {
        std::ifstream f(system_xml);
        if (!f.is_open())
            throw std::runtime_error("Cannot open: " + system_xml);
        system_.reset(OpenMM::XmlSerializer::deserialize<OpenMM::System>(f));
        N = system_->getNumParticles();
    }

    {
        std::ifstream f(integrator_xml);
        if (!f.is_open())
            throw std::runtime_error("Cannot open: " + integrator_xml);
        integrator_.reset(
            OpenMM::XmlSerializer::deserialize<OpenMM::Integrator>(f));
    }

    auto &platform = OpenMM::Platform::getPlatformByName("CPU");
    context_ =
        std::make_unique<OpenMM::Context>(*system_, *integrator_, platform);
}

std::tuple<double, std::vector<double>>
OpenMMSolver::compute(const std::vector<double> &positions) {

    // copy & convert Å → nm (OpenMM internal unit)
    std::vector<OpenMM::Vec3> pos_nm(positions.size() / 3);
#pragma omp parallel for
    for (size_t i = 0; i < pos_nm.size(); ++i) {
        pos_nm[i] = OpenMM::Vec3(
            positions[3 * i + 0] / meika::unit::NANOMETER2ANGSTROM,
            positions[3 * i + 1] / meika::unit::NANOMETER2ANGSTROM,
            positions[3 * i + 2] / meika::unit::NANOMETER2ANGSTROM);
    }

    context_->setPositions(pos_nm);
    auto state =
        context_->getState(OpenMM::State::Forces | OpenMM::State::Energy);

    const double energy =
        state.getPotentialEnergy() / meika::unit::EV2KJ; // → eV

    // forces: kJ/mol/nm  →  eV/Å
    const double ffactor =
        1.0 / (meika::unit::NANOMETER2ANGSTROM * meika::unit::EV2KJ);

    const auto &f_vec = state.getForces();
    std::vector<double> forces(positions.size());
#pragma omp parallel for
    for (size_t i = 0; i < pos_nm.size(); ++i) {
        forces[3 * i + 0] = f_vec[i][0] * ffactor;
        forces[3 * i + 1] = f_vec[i][1] * ffactor;
        forces[3 * i + 2] = f_vec[i][2] * ffactor;
    }

    return {energy, forces};
}

PYBIND11_MODULE(libopenmm_solver, m) {
    m.doc() = "OpenMM full-force-field solver via Context";

    py::class_<OpenMMSolver>(m, "OpenMMSolver")
        .def(py::init<const std::string &, const std::string &>(),
             py::arg("system_xml"), py::arg("integrator_xml"),
             "Load system and integrator XML, create CPU Context.")
        .def("compute", &OpenMMSolver::compute,
             py::call_guard<py::gil_scoped_release>(), py::arg("positions"),
             "Set positions (Angstrom), return (energy_eV, forces_eV_per_A).")
        .def_property_readonly("num_atoms", &OpenMMSolver::num_atoms);
}
