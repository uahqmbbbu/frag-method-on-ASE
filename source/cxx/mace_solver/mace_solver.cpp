#include "mace_solver.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

CalcBuffer::CalcBuffer(torch::Device dev, torch::ScalarType tp,
                       int64_t num_classes, torch::Tensor z_map_cpu,
                       float cutoff, int batch_size_in,
                       int64_t max_atoms_per_struc, int64_t max_edges_per_struc)
    : device(dev), dtype(tp), r_max(cutoff), max_batch_size(batch_size_in),
      max_atoms(max_atoms_per_struc * max_batch_size),
      max_edges(max_edges_per_struc * max_batch_size) {

    auto pinned_f64 = torch::TensorOptions()
                          .device(torch::kCPU)
                          .dtype(torch::kFloat64)
                          .pinned_memory(true);
    auto pinned_i64 = torch::TensorOptions()
                          .device(torch::kCPU)
                          .dtype(torch::kInt64)
                          .pinned_memory(true);
    auto pinned_i32 = torch::TensorOptions()
                          .device(torch::kCPU)
                          .dtype(torch::kInt32)
                          .pinned_memory(true);

    auto dev_tp = torch::TensorOptions().device(device).dtype(dtype);
    auto dev_i64 = torch::TensorOptions().device(device).dtype(torch::kInt64);

    // ---- CPU pinned buffers ----
    ptr_cpu.resize(max_batch_size + 1, 0);
    pinned_ptr = torch::empty({max_batch_size + 1}, pinned_i64);
    pinned_batch = torch::empty({max_atoms}, pinned_i64);
    pinned_positions = torch::empty({max_atoms, 3}, pinned_f64);
    pinned_z = torch::empty({max_atoms}, pinned_i32);

    // ---- device tensors ----
    r_max_sq = torch::full({}, cutoff * cutoff, dev_tp);
    z_map = z_map_cpu.to(device);

    ptr = torch::empty({max_batch_size + 1}, dev_i64);
    batch = torch::empty({max_atoms}, dev_i64);
    positions = torch::empty({max_atoms, 3}, dev_tp);
    cell = torch::empty({max_batch_size, 3, 3}, dev_tp);
    node_attrs = torch::empty({max_atoms, num_classes}, dev_tp);
    edge_index = torch::empty({2, max_edges}, dev_i64);
    shifts = torch::empty({max_edges, 3}, dev_tp);
    unit_shifts = torch::empty({max_edges, 3}, dev_tp);
}

MaceSolver::MaceSolver(const std::string &model_path,
                       const std::string &precision,
                       const std::string &device_str, int batch_size) {

    torch::Device dev = (device_str == "cuda") ? torch::kCUDA : torch::kCPU;
    torch::ScalarType tp =
        (precision == "fp64") ? torch::kFloat64 : torch::kFloat32;

    model_ = std::make_unique<torch::jit::Module>(
        torch::jit::load(model_path, torch::kCPU));

    float r_max = model_->attr("r_max").toTensor().item<float>();
    auto z_table = model_->attr("atomic_numbers").toTensor().to(torch::kCPU);
    int64_t num_classes = z_table.size(0);

    auto z_map_cpu = torch::full({119}, -1, torch::kInt64);
    auto z_acc = z_table.accessor<int64_t, 1>();
    for (int64_t i = 0; i < num_classes; ++i) { z_map_cpu[z_acc[i]] = i; }

    model_->eval();
    model_->to(dev, tp);

    buf_ = std::make_unique<CalcBuffer>(dev, tp, num_classes, z_map_cpu, r_max,
                                        batch_size);

    std::cerr << "MaceSolver: r_max=" << r_max
              << ", num_classes=" << num_classes
              << ", batch_size=" << batch_size << ", device=" << device_str
              << ", precision=" << precision << "\n";
}

void MaceSolver::begin_batch() {
    buf_->batch_size = 0;
    buf_->total_atoms = 0;
    buf_->ptr_cpu[0] = 0;
}

int MaceSolver::batch_count() const { return buf_->batch_size; }

bool MaceSolver::push(std::vector<int32_t> &&atomic_numbers,
                      std::vector<double> &&positions) {
    int64_t i = buf_->batch_size;
    int64_t n = static_cast<int64_t>(atomic_numbers.size());
    int64_t begin = buf_->ptr_cpu[i];

    buf_->ptr_cpu[i + 1] = begin + n;
    buf_->total_atoms = buf_->ptr_cpu[i + 1];

    if (buf_->total_atoms > buf_->max_atoms) {
        throw std::runtime_error("MaceSolver::push: total_atoms " +
                                 std::to_string(buf_->total_atoms) +
                                 " exceeds max_atoms " +
                                 std::to_string(buf_->max_atoms));
    }

    int64_t *batch_ptr = buf_->pinned_batch.data_ptr<int64_t>();
    double *pos_ptr = buf_->pinned_positions.data_ptr<double>();
    int32_t *z_ptr = buf_->pinned_z.data_ptr<int32_t>();

    std::fill(batch_ptr + begin, batch_ptr + begin + n, i);
    std::memcpy(pos_ptr + begin * 3, positions.data(), n * 3 * sizeof(double));
    std::memcpy(z_ptr + begin, atomic_numbers.data(), n * sizeof(int32_t));

    buf_->batch_size++;

    return buf_->batch_size >= buf_->max_batch_size;
}

std::vector<MaceOutput> MaceSolver::forward() {
    int64_t bs = buf_->batch_size;
    int64_t N = buf_->total_atoms;
    bool is_cuda = buf_->device.is_cuda();

    // copy ptr to device
    std::memcpy(buf_->pinned_ptr.data_ptr<int64_t>(), buf_->ptr_cpu.data(),
                (bs + 1) * sizeof(int64_t));
    buf_->ptr.slice(0, 0, bs + 1)
        .copy_(buf_->pinned_ptr.slice(0, 0, bs + 1), is_cuda);

    // copy batch / positions to device
    buf_->batch.slice(0, 0, N).copy_(buf_->pinned_batch.slice(0, 0, N),
                                     is_cuda);
    buf_->positions.slice(0, 0, N).copy_(buf_->pinned_positions.slice(0, 0, N),
                                         is_cuda);

    // z → node_attrs
    auto z_t = buf_->pinned_z.slice(0, 0, N).to(buf_->device);
    auto class_idx = buf_->z_map.index_select(0, z_t.to(torch::kInt64));
    buf_->node_attrs.slice(0, 0, N).zero_();
    buf_->node_attrs.slice(0, 0, N).scatter_(
        1, class_idx.unsqueeze(1).to(torch::kInt64), 1.0);

    // cell
    {
        auto pos = buf_->positions.slice(0, 0, N);
        auto max_coord = pos.abs().max().item<double>();
        double cell_size = (max_coord + 1.0) * 5.0 * buf_->r_max;
        auto single = torch::eye(3, buf_->positions.options()) * cell_size;
        buf_->cell.slice(0, 0, bs).copy_(
            single.unsqueeze(0).repeat({bs, 1, 1}));
    }

    // edges
    {
        auto pos = buf_->positions.slice(0, 0, N);
        auto batch = buf_->batch.slice(0, 0, N);

        auto diff = pos.unsqueeze(1) - pos.unsqueeze(0);
        auto rsq = diff.pow_(2).sum(-1);

        auto same = batch.unsqueeze(1) == batch.unsqueeze(0);
        auto not_self = ~torch::eye(N, pos.options().dtype(torch::kBool));
        auto mask = (rsq < buf_->r_max_sq) & same & not_self;

        auto edges = mask.nonzero();
        int64_t E = edges.size(0);
        if (E > buf_->max_edges) {
            throw std::runtime_error("MaceSolver::flush: edge count " +
                                     std::to_string(E) + " exceeds max_edges " +
                                     std::to_string(buf_->max_edges));
        }
        buf_->total_edges = E;
        buf_->edge_index.slice(1, 0, E).copy_(edges.t().slice(1, 0, E));
        buf_->shifts.slice(0, 0, E).zero_();
        buf_->unit_shifts.slice(0, 0, E).zero_();
    }

    // build dict
    c10::Dict<std::string, torch::Tensor> data;
    data.insert("ptr", buf_->ptr.slice(0, 0, bs + 1));
    data.insert("batch", buf_->batch.slice(0, 0, N));
    data.insert("positions",
                buf_->positions.slice(0, 0, N).set_requires_grad(true));
    data.insert("cell", buf_->cell.slice(0, 0, bs));
    data.insert("node_attrs", buf_->node_attrs.slice(0, 0, N));
    data.insert("edge_index", buf_->edge_index.slice(1, 0, buf_->total_edges));
    data.insert("shifts", buf_->shifts.slice(0, 0, buf_->total_edges));
    data.insert("unit_shifts",
                buf_->unit_shifts.slice(0, 0, buf_->total_edges));

    // forward
    auto outputs = model_->forward({data}).toGenericDict();
    auto energy_t =
        outputs.at("energy").toTensor().to(torch::kCPU, torch::kFloat64);
    auto forces_t =
        outputs.at("forces").toTensor().to(torch::kCPU, torch::kFloat64);

    // unpack
    std::vector<MaceOutput> results(bs);
    for (int64_t i = 0; i < bs; ++i) {
        int64_t begin = buf_->ptr_cpu[i];
        int64_t n = buf_->ptr_cpu[i + 1] - begin;

        results[i].energy = energy_t[i].item<double>();

        results[i].forces.resize(static_cast<size_t>(n) * 3);
        std::memcpy(results[i].forces.data(),
                    forces_t.data_ptr<double>() + begin * 3,
                    n * 3 * sizeof(double));
    }

    // reset for next batch
    buf_->batch_size = 0;
    buf_->total_atoms = 0;
    buf_->ptr_cpu[0] = 0;

    return results;
}
