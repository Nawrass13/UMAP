#include "uwot_transform.h"
#include "uwot_simple_wrapper.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace transform_utils {

    // TEMPORARY: Use exact working version from git commit 65abd80
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

        // Transform operation starting

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

                // Raw point data collected

                // Use stored normalization mode from training
                hnsw_utils::NormalizationPipeline::normalize_data_consistent(
                    raw_point, normalized_point, 1, n_dim,
                    model->feature_means, model->feature_stds,
                    model->normalization_mode);

                // Point normalization completed

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
                    // Convert HNSW distance based on metric
                    float distance = pair.first;
                    switch (model->metric) {
                    case UWOT_METRIC_EUCLIDEAN:
                        distance = std::sqrt(std::max(0.0f, distance)); // L2Space returns squared distance
                        break;
                    case UWOT_METRIC_COSINE:
                        // InnerProductSpace returns -inner_product for unit vectors
                        distance = std::max(0.0f, std::min(2.0f, 1.0f + distance));
                        break;
                    case UWOT_METRIC_MANHATTAN:
                        distance = std::max(0.0f, distance); // Direct Manhattan distance
                        break;
                    default:
                        distance = std::max(0.0f, distance);
                        break;
                    }

                    // Check for exact match (distance near zero)
                    if (distance < match_threshold && !exact_match_found) {
                        exact_match_found = true;
                        exact_match_idx = neighbor_idx;
                    }

                    // Fix 4: Base bandwidth on neighbor distances only (remove min_dist dependency)
                    float eps = 1e-6f;
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

                // Neighbor search completed

                // Store neighbor information
                if (nn_indices && nn_distances) {
                    for (size_t k = 0; k < neighbors.size() && k < static_cast<size_t>(model->n_neighbors); k++) {
                        nn_indices[static_cast<size_t>(i) * static_cast<size_t>(model->n_neighbors) + k] = neighbors[k];
                        nn_distances[static_cast<size_t>(i) * static_cast<size_t>(model->n_neighbors) + k] = distances[k];
                    }
                }

                // Calculate safety metrics
                if (!distances.empty()) {
                    float min_distance = *std::min_element(distances.begin(), distances.end());
                    float mean_distance = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();

                    // Fix 5: Confidence score with robust denominator guard
                    if (confidence_score) {
                        const float EPS = 1e-8f;
                        float denom = std::max(EPS, model->p95_neighbor_distance - model->min_neighbor_distance);
                        float normalized_dist = (min_distance - model->min_neighbor_distance) / denom;
                        confidence_score[i] = std::clamp(1.0f - normalized_dist, 0.0f, 1.0f);
                    }

                    // Outlier level assessment
                    if (outlier_level) {
                        if (min_distance <= model->p95_neighbor_distance) {
                            outlier_level[i] = 0; // Normal
                        }
                        else if (min_distance <= model->p99_neighbor_distance) {
                            outlier_level[i] = 1; // Unusual
                        }
                        else if (min_distance <= model->mild_outlier_threshold) {
                            outlier_level[i] = 2; // Mild outlier
                        }
                        else if (min_distance <= model->extreme_outlier_threshold) {
                            outlier_level[i] = 3; // Extreme outlier
                        }
                        else {
                            outlier_level[i] = 4; // No man's land
                        }
                    }

                    // Fix 5: Percentile rank with guarded denominators
                    if (percentile_rank) {
                        const float EPS = 1e-8f;
                        if (min_distance <= model->min_neighbor_distance) {
                            percentile_rank[i] = 0.0f;
                        }
                        else if (min_distance >= model->p99_neighbor_distance) {
                            percentile_rank[i] = 99.0f;
                        }
                        else {
                            float p95_range = std::max(EPS, model->p95_neighbor_distance - model->min_neighbor_distance);
                            if (min_distance <= model->p95_neighbor_distance) {
                                percentile_rank[i] = 95.0f * (min_distance - model->min_neighbor_distance) / p95_range;
                            }
                            else {
                                float p99_range = std::max(EPS, model->p99_neighbor_distance - model->p95_neighbor_distance);
                                percentile_rank[i] = 95.0f + 4.0f * (min_distance - model->p95_neighbor_distance) / p99_range;
                            }
                        }
                    }

                    // Fix 5: Z-score with robust denominator guard
                    if (z_score) {
                        const float EPS = 1e-8f;
                        float denom_z = std::max(EPS, model->std_neighbor_distance);
                        z_score[i] = (min_distance - model->mean_neighbor_distance) / denom_z;
                    }
                }

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
                    for (int d = 0; d < model->embedding_dim; d++) {
                        float coord = 0.0f;
                        for (size_t k = 0; k < neighbors.size(); k++) {
                            coord += model->embedding[static_cast<size_t>(neighbors[k]) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] * weights[k];
                        }
                        new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] = coord;
                    }
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

    // Minimal uwot_transform that just calls uwot_transform_detailed
    int uwot_transform(
        UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding
    ) {
        return transform_utils::uwot_transform_detailed(model, new_data, n_new_obs, n_dim, embedding,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
}