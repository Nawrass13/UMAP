#pragma once

#include "uwot_model.h"
#include "uwot_hnsw_utils.h"
#include "uwot_progress_utils.h"
#include "uwot_distance.h"
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <atomic>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fit_utils {

    // Import distance metrics from dedicated module
    using namespace distance_metrics;

    // Core k-NN graph building with HNSW optimization
    void build_knn_graph(
        const std::vector<float>& data,
        int n_obs, int n_dim, int n_neighbors,
        UwotMetric metric, UwotModel* model,
        std::vector<int>& nn_indices,
        std::vector<double>& nn_distances,
        int force_exact_knn,
        uwot_progress_callback_v2 progress_callback = nullptr
    );

    // Convert uwot smooth k-NN output to edge list format
    void convert_to_edges(
        const std::vector<int>& nn_indices,
        const std::vector<double>& nn_weights,
        int n_obs, int n_neighbors,
        std::vector<unsigned int>& heads,
        std::vector<unsigned int>& tails,
        std::vector<double>& weights
    );

    // Calculate UMAP parameters from spread and min_dist
    void calculate_ab_from_spread_and_min_dist(UwotModel* model);

    // Compute normalization parameters for training data
    void compute_normalization(
        const std::vector<float>& data,
        int n_obs, int n_dim,
        std::vector<float>& feature_means,
        std::vector<float>& feature_stds
    );

    // Compute neighbor statistics for safety analysis
    void compute_neighbor_statistics(UwotModel* model, const std::vector<float>& normalized_data);

    // Main fit function with progress reporting
    int uwot_fit_with_progress(
        UwotModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float min_dist,
        float spread,
        int n_epochs,
        UwotMetric metric,
        float* embedding,
        uwot_progress_callback progress_callback,
        int force_exact_knn,
        int M,
        int ef_construction,
        int ef_search
    );

    // Enhanced v2 function with loss reporting
    int uwot_fit_with_progress_v2(
        UwotModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float min_dist,
        float spread,
        int n_epochs,
        UwotMetric metric,
        float* embedding,
        uwot_progress_callback_v2 progress_callback,
        int force_exact_knn,
        int M,
        int ef_construction,
        int ef_search
    );
}