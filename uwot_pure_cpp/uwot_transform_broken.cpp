#include "uwot_transform.h"
#include "uwot_simple_wrapper.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace transform_utils {

    namespace detail {
        // Normalize a single data point using stored model parameters
        void normalize_transform_point(
            const std::vector<float>& raw_point,
            std::vector<float>& normalized_point,
            const UwotModel* model
        ) {
            // Use the EXACT same normalization call as the original working version
            std::vector<float> mutable_raw_point = raw_point; // Make a mutable copy for the function
            hnsw_utils::NormalizationPipeline::normalize_data_consistent(
                mutable_raw_point, normalized_point, 1, static_cast<int>(raw_point.size()),
                const_cast<std::vector<float>&>(model->feature_means),
                const_cast<std::vector<float>&>(model->feature_stds),
                const_cast<int&>(model->normalization_mode));
        }

        // Convert HNSW distances to actual metric distances
        float convert_hnsw_distance(float hnsw_dist, UwotMetric metric) {
            switch (metric) {
            case UWOT_METRIC_EUCLIDEAN:
                return std::sqrt(std::max(0.0f, hnsw_dist)); // L2Space returns squared distance
            case UWOT_METRIC_COSINE:
                // InnerProductSpace returns -inner_product for unit vectors
                return std::max(0.0f, std::min(2.0f, 1.0f + hnsw_dist));
            case UWOT_METRIC_MANHATTAN:
                return std::max(0.0f, hnsw_dist); // Direct Manhattan distance
            default:
                return std::max(0.0f, hnsw_dist);
            }
        }

        // Compute weighted interpolation from nearest neighbors
        void compute_weighted_interpolation(
            const std::vector<int>& neighbors,
            const std::vector<float>& weights,
            const UwotModel* model,
            float* result_embedding
        ) {
            // Initialize result to zero
            for (int d = 0; d < model->embedding_dim; d++) {
                result_embedding[d] = 0.0f;
            }

            // Weighted sum of neighbor embeddings
            for (size_t k = 0; k < neighbors.size(); k++) {
                int neighbor_idx = neighbors[k];
                float weight = weights[k];

                for (int d = 0; d < model->embedding_dim; d++) {
                    result_embedding[d] += model->embedding[static_cast<size_t>(neighbor_idx) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] * weight;
                }
            }
        }

        // Calculate safety metrics for transform point
        void calculate_safety_metrics(
            const std::vector<float>& distances,
            const UwotModel* model,
            int point_index,
            float* confidence_score,
            int* outlier_level,
            float* percentile_rank,
            float* z_score
        ) {
            if (distances.empty()) return;

            float min_distance = *std::min_element(distances.begin(), distances.end());

            // Confidence score with robust denominator guard
            if (confidence_score) {
                const float EPS = 1e-8f;
                float denom = std::max(EPS, model->p95_neighbor_distance - model->min_neighbor_distance);
                float normalized_dist = (min_distance - model->min_neighbor_distance) / denom;
                confidence_score[point_index] = std::clamp(1.0f - normalized_dist, 0.0f, 1.0f);
            }

            // Outlier level assessment
            if (outlier_level) {
                if (min_distance <= model->p95_neighbor_distance) {
                    outlier_level[point_index] = 0; // Normal
                }
                else if (min_distance <= model->p99_neighbor_distance) {
                    outlier_level[point_index] = 1; // Unusual
                }
                else if (min_distance <= model->mild_outlier_threshold) {
                    outlier_level[point_index] = 2; // Mild outlier
                }
                else if (min_distance <= model->extreme_outlier_threshold) {
                    outlier_level[point_index] = 3; // Extreme outlier
                }
                else {
                    outlier_level[point_index] = 4; // No man's land
                }
            }

            // Percentile rank with guarded denominators
            if (percentile_rank) {
                const float EPS = 1e-8f;
                if (min_distance <= model->min_neighbor_distance) {
                    percentile_rank[point_index] = 0.0f;
                }
                else if (min_distance >= model->p99_neighbor_distance) {
                    percentile_rank[point_index] = 99.0f;
                }
                else {
                    float p95_range = std::max(EPS, model->p95_neighbor_distance - model->min_neighbor_distance);
                    if (min_distance <= model->p95_neighbor_distance) {
                        percentile_rank[point_index] = 95.0f * (min_distance - model->min_neighbor_distance) / p95_range;
                    }
                    else {
                        float p99_range = std::max(EPS, model->p99_neighbor_distance - model->p95_neighbor_distance);
                        percentile_rank[point_index] = 95.0f + 4.0f * (min_distance - model->p95_neighbor_distance) / p99_range;
                    }
                }
            }

            // Z-score with robust denominator guard
            if (z_score) {
                const float EPS = 1e-8f;
                float denom_z = std::max(EPS, model->std_neighbor_distance);
                z_score[point_index] = (min_distance - model->mean_neighbor_distance) / denom_z;
            }
        }
    }

    // Main transform function for projecting new data points
    int uwot_transform(
        UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding
    ) {
        if (!model || !model->is_fitted || !new_data || !embedding ||
            n_new_obs <= 0 || n_dim != model->n_dim) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        // DEBUG: Print transform state at start
        printf("🔍 DEBUG TRANSFORM - Starting transform:\n");
        printf("  - n_vertices: %d\n", model->n_vertices);
        printf("  - embedding_dim: %d\n", model->embedding_dim);
        printf("  - embedding.size(): %zu\n", model->embedding.size());
        printf("  - mean_neighbor_distance: %.6f\n", model->mean_neighbor_distance);
        printf("  - std_neighbor_distance: %.6f\n", model->std_neighbor_distance);
        printf("  - n_new_obs: %d\n", n_new_obs);
        printf("  - HNSW Settings: M=%d, ef_c=%d, ef_s=%d\n", model->hnsw_M, model->hnsw_ef_construction, model->hnsw_ef_search);
        printf("  - HNSW index exists: %s\n", model->ann_index ? "YES" : "NO");
        if (model->ann_index) {
            printf("  - HNSW index size: %zu elements\n", model->ann_index->getCurrentElementCount());
        }
        printf("  - First 20 embedding points available for transform:\n");
        for (int i = 0; i < std::min(20, model->n_vertices); i++) {
            printf("    Point %d: [%.6f, %.6f]\n", i,
                model->embedding[i * model->embedding_dim],
                model->embedding[i * model->embedding_dim + 1]);
        }

        try {
            std::vector<float> new_embedding(static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim));

            for (int i = 0; i < n_new_obs; i++) {
                // Apply EXACT same normalization as training using unified pipeline
                std::vector<float> raw_point(n_dim);
                std::vector<float> normalized_point;
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    raw_point[j] = new_data[idx];
                }

                // Use stored normalization mode from training - EXACT same pattern as original
                hnsw_utils::NormalizationPipeline::normalize_data_consistent(
                    raw_point, normalized_point, 1, n_dim,
                    model->feature_means, model->feature_stds,
                    model->normalization_mode);

                // CRITICAL SAFETY CHECK: Ensure HNSW index is valid
                if (!model->ann_index) {
                    return UWOT_ERROR_MODEL_NOT_FITTED;
                }

                // Suggestion 1: Boost HNSW search quality for transform
                size_t original_ef = model->ann_index->ef_;
                model->ann_index->setEf(std::max(original_ef, static_cast<size_t>(model->n_neighbors * 4)));  // At least 4x neighbors for 99% recall

                // Use HNSW to find nearest neighbors
                auto search_result = model->ann_index->searchKnn(normalized_point.data(), model->n_neighbors);

                model->ann_index->setEf(original_ef);  // Restore original

                std::vector<int> nn_indices;
                std::vector<float> nn_weights;
                float total_weight = 0.0f;

                // Extract neighbors and compute weights
                bool exact_match_found = false;
                int exact_match_idx = -1;

                // Suggestion 2: Adaptive exact-match threshold
                float match_threshold = 1e-6f * std::sqrt(static_cast<float>(n_dim));

                while (!search_result.empty()) {
                    auto pair = search_result.top();
                    search_result.pop();

                    int neighbor_idx = static_cast<int>(pair.second);
                    float distance = detail::convert_hnsw_distance(pair.first, model->metric);

                    // Check for exact match (distance near zero)
                    if (distance < match_threshold && !exact_match_found) {
                        exact_match_found = true;
                        exact_match_idx = neighbor_idx;
                    }

                    // Fix 4: Use robust neighbor statistics for bandwidth (no min_dist dependency)
                    float median_neighbor_dist = model->median_neighbor_distance > 0.0f ? model->median_neighbor_distance : model->mean_neighbor_distance;
                    float base_bandwidth = std::max(1e-4f, 0.5f * median_neighbor_dist);

                    // For very distant points, increase bandwidth to prevent total weight collapse
                    float adaptive_bandwidth = base_bandwidth;
                    if (distance > base_bandwidth * 2.0f) {
                        adaptive_bandwidth = distance * 0.5f; // Scale bandwidth with distance for distant points
                    }

                    float weight = std::exp(-distance * distance / (2.0f * adaptive_bandwidth * adaptive_bandwidth));

                    // Fix 2: Minimal weight floor to preserve distance sensitivity
                    weight = std::max(weight, 1e-6f);

                    nn_indices.push_back(neighbor_idx);
                    nn_weights.push_back(weight);
                    total_weight += weight;
                }

                // Normalize weights
                if (total_weight > 0.0f) {
                    for (float& w : nn_weights) {
                        w /= total_weight;
                    }
                }

                // CRITICAL FIX: Handle exact matches vs interpolation
                if (exact_match_found) {
                    // Return exact training embedding for exact matches
                    for (int d = 0; d < model->embedding_dim; d++) {
                        new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] =
                            model->embedding[static_cast<size_t>(exact_match_idx) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)];
                    }
                }
                else {
                    // Initialize new point as weighted average of neighbor embeddings
                    float* result_ptr = &new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim)];
                    detail::compute_weighted_interpolation(nn_indices, nn_weights, model, result_ptr);
                }

                // DEBUG: Print detailed calculation for first few points
                if (i < 3) {
                    printf("🔍 DEBUG Point %d transform:\n", i);
                    printf("  - Found %zu neighbors, total_weight: %.6f\n", nn_indices.size(), total_weight);
                    printf("  - First 3 neighbors and weights:\n");
                    for (size_t k = 0; k < std::min(static_cast<size_t>(3), nn_indices.size()); k++) {
                        int idx = nn_indices[k];
                        float weight = nn_weights[k];
                        float emb_x = model->embedding[idx * model->embedding_dim];
                        float emb_y = model->embedding[idx * model->embedding_dim + 1];
                        printf("    Neighbor %zu: idx=%d, weight=%.6f, embedding=[%.6f, %.6f]\n", k, idx, weight, emb_x, emb_y);
                    }
                    printf("  - Final result: [%.6f, %.6f]\n",
                        new_embedding[i * model->embedding_dim],
                        new_embedding[i * model->embedding_dim + 1]);
                }
            }

            // Fix 6: Bounds-checked element-wise copy instead of unsafe memcpy
            size_t expected = static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim);
            if (new_embedding.size() < expected) {
                return UWOT_ERROR_MEMORY;
            }
            for (size_t i = 0; i < expected; ++i) {
                embedding[i] = new_embedding[i];
            }

            return UWOT_SUCCESS;

        }
        catch (...) {
            return UWOT_ERROR_MEMORY;
        }
    }

    // Enhanced transform function with detailed safety metrics (USING EXACT WORKING VERSION)
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
    ) {
        if (!model || !model->is_fitted || !new_data || !embedding ||
            n_new_obs <= 0 || n_dim != model->n_dim) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            std::vector<float> new_embedding(static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim));

            for (int i = 0; i < n_new_obs; i++) {
                // Apply EXACT same normalization as training using unified pipeline
                std::vector<float> raw_point(n_dim);
                std::vector<float> normalized_point;
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    raw_point[j] = new_data[idx];
                }

                // Use stored normalization mode from training - EXACT same pattern as original
                hnsw_utils::NormalizationPipeline::normalize_data_consistent(
                    raw_point, normalized_point, 1, n_dim,
                    model->feature_means, model->feature_stds,
                    model->normalization_mode);

                // CRITICAL SAFETY CHECK: Ensure HNSW index is valid
                if (!model->ann_index) {
                    return UWOT_ERROR_MODEL_NOT_FITTED;
                }

                // FIX #1: Significantly boost ef_search for high-dimensional data with small n_neighbors
                size_t original_ef = model->ann_index->ef_;
                // For high dimensions and small n_neighbors, we need much higher ef_search
                size_t boosted_ef = static_cast<size_t>(model->n_neighbors * 32); // Aggressive boost
                // Cap at a reasonable maximum to avoid excessive computation
                boosted_ef = std::min(boosted_ef, static_cast<size_t>(400));
                model->ann_index->setEf(std::max(original_ef, boosted_ef));

                // Use HNSW to find nearest neighbors
                auto search_result = model->ann_index->searchKnn(normalized_point.data(), model->n_neighbors);

                model->ann_index->setEf(original_ef); // Restore original

                std::vector<int> neighbors;
                std::vector<float> distances;
                std::vector<float> weights;
                float total_weight = 0.0f;

                // Variables for exact match detection
                bool exact_match_found = false;
                int exact_match_idx = -1;

                // Fix 3: Use model's precomputed exact match threshold
                float match_threshold = model->exact_match_threshold;

                // Extract neighbors and compute detailed statistics
                while (!search_result.empty()) {
                    auto pair = search_result.top();
                    search_result.pop();

                    int neighbor_idx = static_cast<int>(pair.second);
                    float distance = detail::convert_hnsw_distance(pair.first, model->metric);

                    // Check for exact match (distance near zero)
                    if (distance < match_threshold && !exact_match_found) {
                        exact_match_found = true;
                        exact_match_idx = neighbor_idx;
                    }

                    // Fix 4: Base bandwidth on neighbor distances only (remove min_dist dependency)
                    float median_neighbor_dist = model->median_neighbor_distance > 0.0f ? model->median_neighbor_distance : model->mean_neighbor_distance;
                    float base_bandwidth = std::max(1e-4f, 0.5f * median_neighbor_dist);

                    // For very distant points, increase bandwidth to prevent total weight collapse
                    float adaptive_bandwidth = base_bandwidth;
                    if (distance > base_bandwidth * 2.0f) {
                        adaptive_bandwidth = distance * 0.3f; // More conservative scaling
                    }

                    float weight = std::exp(-distance * distance / (2.0f * adaptive_bandwidth * adaptive_bandwidth));

                    // Fix 2: Minimal weight floor to preserve distance sensitivity
                    weight = std::max(weight, 1e-6f);

                    neighbors.push_back(neighbor_idx);
                    distances.push_back(distance);
                    weights.push_back(weight);
                    total_weight += weight;
                }

                // Store neighbor information
                if (nn_indices && nn_distances) {
                    for (size_t k = 0; k < neighbors.size() && k < static_cast<size_t>(model->n_neighbors); k++) {
                        nn_indices[static_cast<size_t>(i) * static_cast<size_t>(model->n_neighbors) + k] = neighbors[k];
                        nn_distances[static_cast<size_t>(i) * static_cast<size_t>(model->n_neighbors) + k] = distances[k];
                    }
                }

                // Calculate safety metrics
                detail::calculate_safety_metrics(distances, model, i, confidence_score, outlier_level, percentile_rank, z_score);

                // Handle exact match case (copy training embedding)
                if (exact_match_found) {
                    for (int d = 0; d < model->embedding_dim; d++) {
                        new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] =
                            model->embedding[static_cast<size_t>(exact_match_idx) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)];
                    }
                }
                else {
                    // Normalize weights
                    if (total_weight > 0.0f) {
                        for (float& w : weights) {
                            w /= total_weight;
                        }
                    }

                    // Initialize new point as weighted average of neighbor embeddings
                    float* result_ptr = &new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim)];
                    detail::compute_weighted_interpolation(neighbors, weights, model, result_ptr);
                }
            }

            // Fix 6: Bounds-checked element-wise copy instead of unsafe memcpy
            size_t expected = static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim);
            if (new_embedding.size() < expected) {
                return UWOT_ERROR_MEMORY;
            }
            for (size_t i = 0; i < expected; ++i) {
                embedding[i] = new_embedding[i];
            }

            return UWOT_SUCCESS;

        }
        catch (...) {
            return UWOT_ERROR_MEMORY;
        }
    }

}