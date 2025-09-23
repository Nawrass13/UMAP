#pragma once

#include "uwot_simple_wrapper.h"
#include "uwot_hnsw_utils.h"
#include <memory>
#include <vector>

// Core UMAP model structure
struct UwotModel {
    // Model parameters
    int n_vertices;
    int n_dim;
    int embedding_dim;
    int n_neighbors;
    float min_dist;
    float spread; // UMAP spread parameter (controls global scale)
    UwotMetric metric;
    float a, b; // UMAP curve parameters
    bool is_fitted;
    bool force_exact_knn; // Override flag to force brute-force k-NN

    // HNSW Index for fast neighbor search (replaces training_data storage)
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> ann_index;
    std::unique_ptr<hnsw_utils::SpaceFactory> space_factory;

    // Normalization parameters (moved from C#)
    std::vector<float> feature_means;
    std::vector<float> feature_stds;
    bool use_normalization;
    int normalization_mode;

    // Graph structure using uwot types
    std::vector<unsigned int> positive_head;
    std::vector<unsigned int> positive_tail;
    std::vector<double> positive_weights;  // uwot uses double for weights

    // Final embedding
    std::vector<float> embedding;

    // k-NN structure for transformation (uwot format)
    std::vector<int> nn_indices;      // flattened indices
    std::vector<float> nn_distances;  // flattened distances
    std::vector<float> nn_weights;    // flattened weights for transform

    // Comprehensive neighbor distance statistics for safety detection
    float min_neighbor_distance;
    float mean_neighbor_distance;
    float std_neighbor_distance;
    float p95_neighbor_distance;
    float p99_neighbor_distance;
    float mild_outlier_threshold;      // 2.5 std deviations
    float extreme_outlier_threshold;   // 4.0 std deviations
    float median_neighbor_distance;    // Fix 3: Median distance for robust bandwidth scaling
    float exact_match_threshold;       // Fix 3: Robust exact-match detection threshold

    // HNSW hyperparameters
    int hnsw_M;                        // Graph degree parameter (16-64)
    int hnsw_ef_construction;          // Build quality parameter (64-256)
    int hnsw_ef_search;                // Query quality parameter (32-128)
    bool use_quantization;             // Enable Product Quantization

    // Product Quantization data structures
    std::vector<uint8_t> pq_codes;     // Quantized vector codes (n_vertices * pq_m bytes)
    std::vector<float> pq_centroids;   // PQ codebook (pq_m * 256 * subspace_dim floats)
    int pq_m;                          // Number of subspaces (default: 4)

    UwotModel() : n_vertices(0), n_dim(0), embedding_dim(2), n_neighbors(15),
        min_dist(0.1f), spread(1.0f), metric(UWOT_METRIC_EUCLIDEAN), a(1.929f), b(0.7915f),
        is_fitted(false), force_exact_knn(false), use_normalization(false),
        min_neighbor_distance(0.0f), mean_neighbor_distance(0.0f),
        std_neighbor_distance(0.0f), p95_neighbor_distance(0.0f),
        p99_neighbor_distance(0.0f), mild_outlier_threshold(0.0f),
        extreme_outlier_threshold(0.0f), median_neighbor_distance(0.0f),
        exact_match_threshold(1e-3f), hnsw_M(32), hnsw_ef_construction(128),
        hnsw_ef_search(64), use_quantization(true), pq_m(4), normalization_mode(1) {

        space_factory = std::make_unique<hnsw_utils::SpaceFactory>();
    }
};

// Model lifecycle functions
namespace model_utils {
    UwotModel* create_model();
    void destroy_model(UwotModel* model);

    // Model information functions
    int get_model_info(UwotModel* model, int* n_vertices, int* n_dim, int* embedding_dim,
        int* n_neighbors, float* min_dist, float* spread, UwotMetric* metric,
        int* hnsw_M, int* hnsw_ef_construction, int* hnsw_ef_search);

    // Utility functions
    int get_embedding_dim(UwotModel* model);
    int get_n_vertices(UwotModel* model);
    int is_fitted(UwotModel* model);

    // Error handling
    const char* get_error_message(int error_code);
    const char* get_metric_name(UwotMetric metric);
    const char* get_version();
}