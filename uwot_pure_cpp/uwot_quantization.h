#pragma once

#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace pq_utils {

    // Calculate optimal number of PQ subspaces (pq_m) for given dimension
    // Finds largest divisor of n_dim that creates subspaces with ≥ min_subspace_dim dimensions
    // Prioritizes common values (2, 4, 8, 16) for optimal performance
    int calculate_optimal_pq_m(int n_dim, int min_subspace_dim = 4);

    // Simple k-means clustering for PQ codebook generation
    void simple_kmeans(
        const std::vector<float>& data,
        int n_points,
        int dim,
        int k,
        std::vector<float>& centroids,
        std::vector<int>& assignments
    );

    // Perform Product Quantization encoding
    void encode_pq(
        const std::vector<float>& data,
        int n_points,
        int dim,
        int m,
        std::vector<uint8_t>& codes,
        std::vector<float>& centroids
    );

    // Reconstruct vector from PQ codes
    void reconstruct_vector(
        const std::vector<uint8_t>& codes,
        int point_idx,
        int m,
        const std::vector<float>& centroids,
        int subspace_dim,
        std::vector<float>& reconstructed
    );

}