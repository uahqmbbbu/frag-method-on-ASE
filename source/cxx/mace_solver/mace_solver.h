#pragma once

#include <torch/script.h>

#include <memory>
#include <string>
#include <vector>

struct MaceOutput {
    double energy;
    std::vector<double> forces;
};

struct CalcBuffer {
    torch::Device device;
    torch::ScalarType dtype;

    float r_max;
    torch::Tensor r_max_sq;
    torch::Tensor z_map;

    // size limits
    int64_t max_batch_size = 0;
    int64_t max_atoms = 0;
    int64_t max_edges = 0;

    // current batch dimensions
    int64_t batch_size = 0;
    int64_t total_atoms = 0;
    int64_t total_edges = 0;

    // ---- CPU pinned buffers (filled directly by caller) ----
    std::vector<int64_t> ptr_cpu;   // [max_batch_size+1]  plain vector
    torch::Tensor pinned_ptr;       // [max_batch_size+1]  int64 pinned
    torch::Tensor pinned_batch;     // [max_atoms]         int64 pinned
    torch::Tensor pinned_positions; // [max_atoms, 3]      f64 pinned
    torch::Tensor pinned_z;         // [max_atoms]         int32 pinned

    // ---- device tensors (CUDA or CPU, depending on device) ----
    torch::Tensor ptr;         // [max_batch_size + 1]
    torch::Tensor batch;       // [max_atoms]
    torch::Tensor positions;   // [max_atoms, 3]   (fp32 or fp64)
    torch::Tensor cell;        // [max_batch_size, 3, 3]
    torch::Tensor node_attrs;  // [max_atoms, num_classes]
    torch::Tensor edge_index;  // [2, max_edges]
    torch::Tensor shifts;      // [max_edges, 3]
    torch::Tensor unit_shifts; // [max_edges, 3]

    CalcBuffer(torch::Device dev, torch::ScalarType tp, int64_t num_classes,
               torch::Tensor z_map_cpu, float cutoff, int batch_size_in,
               int64_t max_atoms_per_struc = 150,
               int64_t max_edges_per_struc = 8000);
};

class MaceSolver {
  public:
    MaceSolver(const std::string &model_path,
               const std::string &precision = "fp32",
               const std::string &device_str = "cpu", int batch_size = 128);

    int batch_count() const;
    void begin_batch();

    bool push(std::vector<int32_t> &&atomic_numbers,
              std::vector<double> &&positions);
    std::vector<MaceOutput> forward();

  private:
    std::unique_ptr<torch::jit::Module> model_;
    std::unique_ptr<CalcBuffer> buf_;
};
