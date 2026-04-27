#include "meika/pdb.h"

#include "eegmfcc_solver.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <vector>

EEGMFCCSolver::EEGMFCCSolver(const std::string &pdb_file,
                             const std::string &model_path,
                             const std::string &precision,
                             const std::string &device, int batch_size,
                             const std::string &system_xml)
    : sys_(meika::pdb::readPDB(pdb_file)) {
    N = sys_.n;
    build_atomic_numbers();
    build_fragments();
    build_exclude_pairs();

    mace_solver_ =
        std::make_unique<MaceSolver>(model_path, precision, device, batch_size);
    if (!system_xml.empty()) {
        nb_solver_ =
            std::make_unique<NonBondedSolver>(system_xml, exclude_pairs_);
    }
}

void EEGMFCCSolver::build_atomic_numbers() {
    const auto &elem_map = meika::system::get_name_to_atomic_number_map();
    atomic_numbers_.resize(N);
    for (size_t i = 0; i < N; ++i) {
        auto it = elem_map.find(sys_.element[i]);
        atomic_numbers_[i] = (it != elem_map.end()) ? it->second : 0;
    }
}

std::vector<std::vector<int>> EEGMFCCSolver::frag2atom() {
    const auto ter_res_set =
        std::set<int>(sys_.ter_res.begin(), sys_.ter_res.end());
    const size_t num_residues =
        *(std::max_element(sys_.res_idx.begin(), sys_.res_idx.end())) + 1;

    std::vector<std::vector<int>> frag_atom(num_residues);
    for (size_t i = 0; i < sys_.res_idx.size(); ++i) {
        auto &res = sys_.res_idx[i];
        if (sys_.name[i] == "O3'" and ter_res_set.count(sys_.res_idx[i]) == 0) {
            res++;
            sys_.name[i] = "OP3";
        }
        frag_atom[res].emplace_back(i);
    }
    return frag_atom;
}

std::vector<std::pair<int, int>> EEGMFCCSolver::split_chain() const {
    std::vector<std::pair<int, int>> output;
    for (size_t i = 0; i < sys_.ter_res.size(); ++i) {
        auto chain_start = i == 0 ? sys_.res_idx[0] : sys_.ter_res[i - 1] + 1;
        auto chain_ter = sys_.ter_res[i];
        output.emplace_back(chain_start, chain_ter);
    }
    return output;
}

void EEGMFCCSolver::build_fragments() {
    auto frag_atom = frag2atom();
    auto regions = split_chain();

    constexpr double OHLength = 0.98;
    constexpr double CHLength = 1.09;

    auto find_atom_in_res = [this, &frag_atom](int ridx,
                                               const std::string &name) -> int {
        for (int ai : frag_atom[ridx]) {
            if (sys_.name[ai] == name) return ai;
        }
        throw std::runtime_error("Atom '" + name + "' not found in residue " +
                                 std::to_string(ridx));
    };

    for (size_t i = 0; i < regions.size(); ++i) {
        const auto res_begin = regions[i].first;
        const auto res_end = regions[i].second;

        // cap-body (+1)
        for (int ii = res_begin + 1; ii <= res_end - 1; ++ii) {
            FragInfo frag;
            frag.sign = +1;

            for (int ri : {ii - 1, ii, ii + 1}) {
                for (int ai : frag_atom[ri]) { frag.atoms.push_back(ai); }
            }

            if (ii != res_begin + 1) {
                frag.ghosts.push_back({find_atom_in_res(ii - 1, "OP3"),
                                       {find_atom_in_res(ii - 2, "C3'")},
                                       OHLength});
            }
            if (ii != res_end - 1) {
                frag.ghosts.push_back({find_atom_in_res(ii + 1, "C3'"),
                                       {find_atom_in_res(ii + 2, "OP3")},
                                       CHLength});
            }

            fragments_.push_back(std::move(frag));
        }

        // concap (-1)
        for (int ii = res_begin + 1; ii <= res_end - 2; ++ii) {
            FragInfo frag;
            frag.sign = -1;

            for (int ri : {ii, ii + 1}) {
                for (int ai : frag_atom[ri]) { frag.atoms.push_back(ai); }
            }

            frag.ghosts.push_back({find_atom_in_res(ii, "OP3"),
                                   {find_atom_in_res(ii - 1, "C3'")},
                                   OHLength});
            frag.ghosts.push_back({find_atom_in_res(ii + 1, "C3'"),
                                   {find_atom_in_res(ii + 2, "OP3")},
                                   CHLength});

            fragments_.push_back(std::move(frag));
        }
    }
}

void EEGMFCCSolver::build_exclude_pairs() {
    std::map<std::pair<int, int>, int> exclude_count;
    for (const auto &frag : fragments_) {
        for (size_t a = 0; a < frag.atoms.size(); ++a) {
            for (size_t b = a + 1; b < frag.atoms.size(); ++b) {
                int i = frag.atoms[a];
                int j = frag.atoms[b];
                if (i > j) std::swap(i, j);
                exclude_count[{i, j}] += frag.sign;
            }
        }
    }
    for (const auto &kv : exclude_count) {
        if (kv.second > 0) { exclude_pairs_.push_back(kv.first); }
    }
}

std::tuple<std::vector<int32_t>, std::vector<double>>
EEGMFCCSolver::pre_model_input(const FragInfo &frag, const double *pos) {
    std::vector<int32_t> z;
    std::vector<double> coord;

    for (int gi : frag.atoms) {
        z.push_back(atomic_numbers_[gi]);
        coord.push_back(pos[3 * gi + 0]);
        coord.push_back(pos[3 * gi + 1]);
        coord.push_back(pos[3 * gi + 2]);
    }

    for (const auto &gh : frag.ghosts) {
        double dx = 0, dy = 0, dz = 0;
        for (int ci : gh.cons) {
            dx += pos[3 * ci + 0] - pos[3 * gh.root + 0];
            dy += pos[3 * ci + 1] - pos[3 * gh.root + 1];
            dz += pos[3 * ci + 2] - pos[3 * gh.root + 2];
        }
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        double hx = pos[3 * gh.root + 0] + (dx / dist) * gh.bond_length;
        double hy = pos[3 * gh.root + 1] + (dy / dist) * gh.bond_length;
        double hz = pos[3 * gh.root + 2] + (dz / dist) * gh.bond_length;

        z.push_back(1);
        coord.push_back(hx);
        coord.push_back(hy);
        coord.push_back(hz);
    }

    return {z, coord};
}

std::tuple<double, py::array_t<double>>
EEGMFCCSolver::compute(py::array_t<double> positions_py) {
    auto buf_pos = positions_py.request();
    const double *pos = static_cast<const double *>(buf_pos.ptr);
    size_t n_atoms = buf_pos.shape[0];

    py::array_t<double> forces_arr(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(n_atoms), 3});
    double *force_ptr = forces_arr.mutable_data();
    std::memset(force_ptr, 0, n_atoms * 3 * sizeof(double));

    double energy = 0.0;

    {
        py::gil_scoped_release unlock;

        // ---- QM (MACE EEGMFCC) ----
        mace_solver_->begin_batch();
        size_t chunk_start = 0;

        for (size_t fi = 0; fi < fragments_.size(); ++fi) {
            const auto &frag = fragments_[fi];
            auto [z, coord] = pre_model_input(frag, pos);
            // dump_fragments(z, coord);

            bool full = mace_solver_->push(std::move(z), std::move(coord));

            if (full || fi == fragments_.size() - 1) {
                auto results = mace_solver_->forward();

                for (size_t ri = 0; ri < results.size(); ++ri) {
                    const auto &rfrag = fragments_[chunk_start + ri];
                    int sign = rfrag.sign;
                    energy += sign * results[ri].energy;

                    for (size_t k = 0; k < rfrag.atoms.size() * 3; ++k) {
                        int gi = rfrag.atoms[k / 3];
                        force_ptr[3 * gi + k % 3] +=
                            sign * results[ri].forces[k];
                    }
                }
                chunk_start = fi + 1;
            }
        }

        // ---- MM nonbonded ----
        if (nb_solver_) energy += nb_solver_->compute(pos, force_ptr);
    } // gil_scoped_release

    return {energy, forces_arr};
}

void EEGMFCCSolver::dump_fragments(const std::vector<int32_t> &z,
                                   const std::vector<double> &coord) const {
    static const char *elem[] = {"?",  "H", "He", "Li", "Be", "B",  "C",
                                 "N",  "O", "F",  "Ne", "Na", "Mg", "Al",
                                 "Si", "P", "S",  "Cl", "Ar"};
    static int counter = 0;
    static const std::string dir = "/home/jiabao/tmp/fragments";
    static bool inited = []() {
        std::filesystem::create_directories(dir);
        return true;
    }();

    std::ostringstream ss;
    ss << dir << "/fragment_" << std::setfill('0') << std::setw(4) << counter++
       << ".xyz";
    std::ofstream f(ss.str());
    if (!f) return;
    size_t n = z.size();
    f << n << "\n# fragment\n";
    for (size_t i = 0; i < n; ++i) {
        int zz = z[i];
        const char *name = (zz >= 0 && zz < 19) ? elem[zz] : "X";
        std::ostringstream line;
        line << std::setw(2) << name;
        line << std::fixed << std::setprecision(6);
        line << std::setw(13) << coord[3 * i + 0];
        line << std::setw(13) << coord[3 * i + 1];
        line << std::setw(13) << coord[3 * i + 2] << "\n";
        f << line.str();
    }
}

PYBIND11_MODULE(libeegmfcc_solver, m) {
    m.doc() = "EEGMFCC-style QM solver with pybind11 bindings";

    py::class_<EEGMFCCSolver>(m, "EEGMFCCSolver")
        .def(py::init<const std::string &, const std::string &,
                      const std::string &, const std::string &, int,
                      const std::string &>(),
             py::arg("pdb_file"), py::arg("model_path"),
             py::arg("precision") = "fp32", py::arg("device") = "cpu",
             py::arg("batch_size") = 64, py::arg("system_xml") = "",
             "Initialise from a PDB file and MACE model.")
        .def("compute", &EEGMFCCSolver::compute, py::arg("positions"),
             "Compute QM (MACE) + MM (nonbonded) energy and forces.");
}
