#include "eegmfcc_solver.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <vector>

#include <pybind11/stl.h>

#include "meika/pdb.h"

// ===================================================================
//  Constructor
// ===================================================================

EEGMFCCSolver::EEGMFCCSolver(const std::string &pdb_file,
                             const std::string &model_path,
                             const std::string &precision,
                             const std::string &device, int batch_size)
    : sys_(meika::pdb::readPDB(pdb_file)) {
    n_atoms_ = sys_.n;
    map_name_to_z();
    pre_frag();
    solver_ =
        std::make_unique<MaceSolver>(model_path, precision, device, batch_size);
}

// ===================================================================
//  Init helpers
// ===================================================================

void EEGMFCCSolver::map_name_to_z() {
    const auto &elem_map = meika::system::get_name_to_atomic_number_map();
    atomic_numbers_.resize(n_atoms_);
    for (int i = 0; i < n_atoms_; ++i) {
        auto it = elem_map.find(sys_.element[i]);
        atomic_numbers_[i] = (it != elem_map.end()) ? it->second : 0;
    }
}

std::vector<std::vector<int>> EEGMFCCSolver::frag2atom() {
    const auto ter_res_set =
        std::set<int>(sys_.ter_res.begin(), sys_.ter_res.end());
    const std::size_t num_residues =
        *(std::max_element(sys_.res_idx.begin(), sys_.res_idx.end())) + 1;

    std::vector<std::vector<int>> frag_atom(num_residues);
    for (std::size_t i = 0; i < sys_.res_idx.size(); ++i) {
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
    for (std::size_t i = 0; i < sys_.ter_res.size(); ++i) {
        auto chain_start = i == 0 ? sys_.res_idx[0] : sys_.ter_res[i - 1] + 1;
        auto chain_ter = sys_.ter_res[i];
        output.emplace_back(chain_start, chain_ter);
    }
    return output;
}

void EEGMFCCSolver::pre_frag() {
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

    for (std::size_t i = 0; i < regions.size(); ++i) {
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

// ===================================================================
//  compute() — push fragments directly into solver pinned memory
// ===================================================================

std::tuple<double, py::array_t<double>>
EEGMFCCSolver::compute(py::array_t<int32_t> /*atomic_numbers_py*/,
                       py::array_t<double> positions_py) {
    auto buf_pos = positions_py.request();
    const double *pos = static_cast<const double *>(buf_pos.ptr);
    size_t n_atoms = buf_pos.shape[0];

    py::array_t<double> forces_arr(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(n_atoms), 3});
    double *force_ptr = forces_arr.mutable_data();
    std::memset(force_ptr, 0, n_atoms * 3 * sizeof(double));

    double total_energy = 0.0;

    solver_->begin_batch();
    size_t chunk_start = 0; // fragment index where current batch starts

    for (size_t fi = 0; fi < fragments_.size(); ++fi) {
        const auto &frag = fragments_[fi];

        // ---- gather z & coord for this fragment (real + ghost) ----
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

            z.push_back(1); // H
            coord.push_back(hx);
            coord.push_back(hy);
            coord.push_back(hz);
        }

        // ---- push directly into solver's pinned memory ----
        bool full = solver_->push(std::move(z), std::move(coord));

        if (full || fi == fragments_.size() - 1) {
            auto results = solver_->flush();

            for (size_t ri = 0; ri < results.size(); ++ri) {
                const auto &rfrag = fragments_[chunk_start + ri];
                int sign = rfrag.sign;
                total_energy += sign * results[ri].energy;

                for (size_t k = 0; k < rfrag.atoms.size(); ++k) {
                    int gi = rfrag.atoms[k];
                    force_ptr[3 * gi + 0] +=
                        sign * results[ri].forces[3 * k + 0];
                    force_ptr[3 * gi + 1] +=
                        sign * results[ri].forces[3 * k + 1];
                    force_ptr[3 * gi + 2] +=
                        sign * results[ri].forces[3 * k + 2];
                }
            }
            chunk_start = fi + 1;
        }
    }

    return {total_energy, forces_arr};
}

// ===================================================================
//  pybind11 module
// ===================================================================

PYBIND11_MODULE(libeegmfcc_solver, m) {
    m.doc() = "EEGMFCC-style QM solver with pybind11 bindings";

    py::class_<EEGMFCCSolver>(m, "EEGMFCCSolver")
        .def(py::init<const std::string &, const std::string &,
                      const std::string &, const std::string &, int>(),
             py::arg("pdb_file"), py::arg("model_path"),
             py::arg("precision") = "fp32", py::arg("device") = "cpu",
             py::arg("batch_size") = 64,
             "Initialise from a PDB file and MACE model.")
        .def("compute", &EEGMFCCSolver::compute, py::arg("atomic_numbers"),
             py::arg("positions"),
             "Compute QM energy and forces.\n\n"
             "Args:\n"
             "    atomic_numbers: ndarray (n_atoms,) int32\n"
             "    positions:      ndarray (n_atoms, 3) float64\n"
             "Returns:\n"
             "    energy: float\n"
             "    forces: ndarray (n_atoms, 3) float64\n");
}
