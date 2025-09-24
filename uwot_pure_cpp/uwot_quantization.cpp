#include "uwot_quantization.h"
#include <iostream>
#include <algorithm>
#include <numeric>

namespace pq_utils {

    // Calculate optimal number of PQ subspaces (pq_m) for given dimension
    // Finds largest divisor of n_dim that creates subspaces with ≥ min_subspace_dim dimensions
    // Prioritizes common values (2, 4, 8, 16) for optimal performance
    int calculate_optimal_pq_m(int n_dim, int min_subspace_dim) {
        if (n_dim < min_subspace_dim) {
            return 1; // Disable PQ for very small dimensions
        }

        // Preferred pq_m values in order of preference (larger = more compression)
        std::vector<int> preferred_values = { 16, 8, 4, 2 };

        for (int pq_m : preferred_values) {
            if (n_dim % pq_m == 0 && (n_dim / pq_m) >= min_subspace_dim) {
                return pq_m;
            }
        }

        // If preferred values don't work, find largest suitable divisor
        for (int pq_m = n_dim / min_subspace_dim; pq_m >= 1; pq_m--) {
            if (n_dim % pq_m == 0) {
                return pq_m;
            }
        }

        return 1; // Fallback: disable PQ
    }

    // Simple k-means clustering for PQ codebook generation
    void simple_kmeans(const std::vector<float>& data, int n_points, int dim, int k,
        std::vector<float>& centroids, std::vector<int>& assignments) {
        centroids.resize(k * dim);
        assignments.resize(n_points);

        // Initialize centroids deterministically for reproducible results
        std::mt19937 gen(42);
        std::uniform_int_distribution<> dis(0, n_points - 1);

        for (int c = 0; c < k; c++) {
            int random_idx = dis(gen);
            for (int d = 0; d < dim; d++) {
                centroids[c * dim + d] = data[random_idx * dim + d];
            }
        }

        // Iterate k-means with convergence checking and adaptive iterations
        const int MAX_ITERATIONS = 100; // Increased from 25
        const float CONVERGENCE_EPSILON = 1e-6f; // Convergence threshold

        for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
            // Assignment step
            for (int i = 0; i < n_points; i++) {
                float min_dist = std::numeric_limits<float>::max();
                int best_cluster = 0;

                for (int c = 0; c < k; c++) {
                    float dist = 0.0f;
                    for (int d = 0; d < dim; d++) {
                        float diff = data[i * dim + d] - centroids[c * dim + d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_cluster = c;
                    }
                }
                assignments[i] = best_cluster;
            }

            // Update step with convergence tracking
            std::vector<int> counts(k, 0);
            std::vector<float> new_centroids(k * dim, 0.0f);

            for (int i = 0; i < n_points; i++) {
                int c = assignments[i];
                counts[c]++;
                for (int d = 0; d < dim; d++) {
                    new_centroids[c * dim + d] += data[i * dim + d];
                }
            }

            // Calculate new centroids and detect convergence
            float total_change = 0.0f;
            std::uniform_int_distribution<> reinit_dis(0, n_points - 1);

            for (int c = 0; c < k; c++) {
                if (counts[c] > 0) {
                    // Normal update for non-empty clusters
                    for (int d = 0; d < dim; d++) {
                        float old_val = centroids[c * dim + d];
                        float new_val = new_centroids[c * dim + d] / counts[c];
                        centroids[c * dim + d] = new_val;

                        // Track change for convergence
                        total_change += (new_val - old_val) * (new_val - old_val);
                    }
                } else {
                    // Re-initialize empty clusters with random points
                    int random_idx = reinit_dis(gen);
                    for (int d = 0; d < dim; d++) {
                        centroids[c * dim + d] = data[random_idx * dim + d];
                    }
                    total_change += 1.0f; // Prevent early convergence
                }
            }

            // Check for convergence
            if (total_change < CONVERGENCE_EPSILON && iter > 0) {
                break; // Converged, stop iterations
            }
        }
    }

    // Perform Product Quantization encoding
    void encode_pq(const std::vector<float>& data, int n_points, int dim, int m,
        std::vector<uint8_t>& codes, std::vector<float>& centroids) {
        if (dim % m != 0) {
            throw std::runtime_error("Dimension must be divisible by number of subspaces");
        }

        int subspace_dim = dim / m;
        int num_centroids = 256; // 8-bit codes

        codes.resize(n_points * m);
        centroids.resize(m * num_centroids * subspace_dim);

        // Process each subspace
        for (int sub = 0; sub < m; sub++) {
            // Extract subspace data
            std::vector<float> subspace_data(n_points * subspace_dim);
            for (int i = 0; i < n_points; i++) {
                for (int d = 0; d < subspace_dim; d++) {
                    subspace_data[i * subspace_dim + d] = data[i * dim + sub * subspace_dim + d];
                }
            }

            // Run k-means for this subspace
            std::vector<float> sub_centroids;
            std::vector<int> assignments;
            simple_kmeans(subspace_data, n_points, subspace_dim, num_centroids, sub_centroids, assignments);

            // Store centroids for this subspace
            for (int c = 0; c < num_centroids; c++) {
                for (int d = 0; d < subspace_dim; d++) {
                    centroids[sub * num_centroids * subspace_dim + c * subspace_dim + d] = sub_centroids[c * subspace_dim + d];
                }
            }

            // Store codes for this subspace
            for (int i = 0; i < n_points; i++) {
                codes[i * m + sub] = static_cast<uint8_t>(assignments[i]);
            }
        }
    }

    // Reconstruct vector from PQ codes
    void reconstruct_vector(const std::vector<uint8_t>& codes, int point_idx, int m,
        const std::vector<float>& centroids, int subspace_dim,
        std::vector<float>& reconstructed) {
        reconstructed.resize(m * subspace_dim);

        for (int sub = 0; sub < m; sub++) {
            uint8_t code = codes[point_idx * m + sub];
            int centroid_offset = sub * 256 * subspace_dim + code * subspace_dim;

            for (int d = 0; d < subspace_dim; d++) {
                reconstructed[sub * subspace_dim + d] = centroids[centroid_offset + d];
            }
        }
    }

}