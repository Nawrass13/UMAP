#pragma once

#include "uwot_model.h"
#include "uwot_hnsw_utils.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace transform_utils {

    // Main transform function for projecting new data points
    int uwot_transform(
        UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding
    );

    // Enhanced transform function with detailed safety metrics
    int uwot_transform_detailed(
        UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding,
        int* nn_indices,
        float* nn_distances,
        float* confidence_score,
        int* outlier_level,
        float* percentile_rank,
        float* z_score
    );

    // Helper functions for transform operations
    namespace detail {
        // Normalize a single data point using stored model parameters
        void normalize_transform_point(
            const std::vector<float>& raw_point,
            std::vector<float>& normalized_point,
            const UwotModel* model
        );

        // Compute weighted interpolation from nearest neighbors
        void compute_weighted_interpolation(
            const std::vector<int>& neighbors,
            const std::vector<float>& weights,
            const UwotModel* model,
            float* result_embedding
        );

        // Calculate safety metrics for transform point
        void calculate_safety_metrics(
            const std::vector<float>& distances,
            const UwotModel* model,
            int point_index,
            float* confidence_score,
            int* outlier_level,
            float* percentile_rank,
            float* z_score
        );

        // Convert HNSW distances to actual metric distances
        float convert_hnsw_distance(float hnsw_dist, UwotMetric metric);
    }
}