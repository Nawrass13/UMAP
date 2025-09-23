#include "uwot_fit.h"
#include "uwot_simple_wrapper.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <limits>

// Include uwot headers
#include "smooth_knn.h"

namespace fit_utils {

    // Helper function to compute normalization parameters
    void compute_normalization(const std::vector<float>& data, int n_obs, int n_dim,
        std::vector<float>& means, std::vector<float>& stds) {
        means.resize(n_dim);
        stds.resize(n_dim);

        // Calculate means
        std::fill(means.begin(), means.end(), 0.0f);
        for (int i = 0; i < n_obs; i++) {
            for (int j = 0; j < n_dim; j++) {
                means[j] += data[static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j)];
            }
        }
        for (int j = 0; j < n_dim; j++) {
            means[j] /= static_cast<float>(n_obs);
        }

        // Calculate standard deviations
        std::fill(stds.begin(), stds.end(), 0.0f);
        for (int i = 0; i < n_obs; i++) {
            for (int j = 0; j < n_dim; j++) {
                float diff = data[static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j)] - means[j];
                stds[j] += diff * diff;
            }
        }
        for (int j = 0; j < n_dim; j++) {
            stds[j] = std::sqrt(stds[j] / static_cast<float>(n_obs - 1));
            if (stds[j] < 1e-8f) stds[j] = 1.0f; // Prevent division by zero
        }
    }

    // Helper function to compute comprehensive neighbor statistics
    void compute_neighbor_statistics(UwotModel* model, const std::vector<float>& normalized_data) {
        if (!model->ann_index || model->n_vertices == 0) return;

        std::vector<float> all_distances;
        all_distances.reserve(model->n_vertices * model->n_neighbors);

        // Query each point to get neighbor distances
        for (int i = 0; i < model->n_vertices; i++) {
            const float* query_point = &normalized_data[static_cast<size_t>(i) * static_cast<size_t>(model->n_dim)];

            try {
                // Search for k+1 neighbors (includes self)
                auto result = model->ann_index->searchKnn(query_point, model->n_neighbors + 1);

                // Skip the first result (self) and collect distances
                int count = 0;
                while (!result.empty() && count < model->n_neighbors) {
                    auto pair = result.top();
                    result.pop();

                    if (count > 0) { // Skip self-distance
                        // Convert HNSW distance based on metric
                        float distance = pair.first;
                        switch (model->metric) {
                        case UWOT_METRIC_EUCLIDEAN:
                            distance = std::sqrt(std::max(0.0f, distance)); // L2Space returns squared distance
                            break;
                        case UWOT_METRIC_COSINE:
                            // InnerProductSpace returns -inner_product for unit vectors
                            // Convert to cosine distance: distance = 1 - similarity
                            distance = std::max(0.0f, std::min(2.0f, 1.0f + distance));
                            break;
                        case UWOT_METRIC_MANHATTAN:
                            // L1Space returns direct Manhattan distance
                            distance = std::max(0.0f, distance);
                            break;
                        default:
                            distance = std::max(0.0f, distance);
                            break;
                        }
                        all_distances.push_back(distance);
                    }
                    count++;
                }
            }
            catch (...) {
                // Handle any HNSW exceptions gracefully
                continue;
            }
        }

        if (all_distances.empty()) return;

        // Sort distances for percentile calculations
        std::sort(all_distances.begin(), all_distances.end());

        // Calculate statistics
        model->min_neighbor_distance = all_distances.front();

        // Mean calculation
        float sum = 0.0f;
        for (float dist : all_distances) {
            sum += dist;
        }
        model->mean_neighbor_distance = sum / all_distances.size();

        // Standard deviation calculation
        float sq_sum = 0.0f;
        for (float dist : all_distances) {
            float diff = dist - model->mean_neighbor_distance;
            sq_sum += diff * diff;
        }
        model->std_neighbor_distance = std::sqrt(sq_sum / all_distances.size());

        // Percentile calculations
        size_t p95_idx = static_cast<size_t>(0.95 * all_distances.size());
        size_t p99_idx = static_cast<size_t>(0.99 * all_distances.size());
        model->p95_neighbor_distance = all_distances[std::min(p95_idx, all_distances.size() - 1)];
        model->p99_neighbor_distance = all_distances[std::min(p99_idx, all_distances.size() - 1)];

        // Fix 3: Compute median neighbor distance
        size_t median_idx = all_distances.size() / 2;
        model->median_neighbor_distance = all_distances[median_idx];

        // Fix 3: Set robust exact-match threshold for float32
        model->exact_match_threshold = 1e-3f / std::sqrt(static_cast<float>(model->n_dim));

        // Outlier thresholds
        model->mild_outlier_threshold = model->mean_neighbor_distance + 2.5f * model->std_neighbor_distance;
        model->extreme_outlier_threshold = model->mean_neighbor_distance + 4.0f * model->std_neighbor_distance;

        printf("[STATS] Neighbor distances - min: %.4f, median: %.4f, mean: %.4f +/- %.4f, p95: %.4f, p99: %.4f\n",
            model->min_neighbor_distance, model->median_neighbor_distance, model->mean_neighbor_distance,
            model->std_neighbor_distance, model->p95_neighbor_distance, model->p99_neighbor_distance);
        printf("[STATS] Outlier thresholds - mild: %.4f, extreme: %.4f\n",
            model->mild_outlier_threshold, model->extreme_outlier_threshold);
    }

    // Distance metric implementations

    // Build k-NN graph using specified distance metric
    void build_knn_graph(const std::vector<float>& data, int n_obs, int n_dim,
        int n_neighbors, UwotMetric metric, UwotModel* model,
        std::vector<int>& nn_indices, std::vector<double>& nn_distances,
        int force_exact_knn, uwot_progress_callback_v2 progress_callback) {

        nn_indices.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));
        nn_distances.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));

        auto start_time = std::chrono::steady_clock::now();

        // Enhanced HNSW optimization check with model availability
        bool can_use_hnsw = !force_exact_knn &&
            model && model->space_factory && model->space_factory->can_use_hnsw() &&
            model->ann_index && model->ann_index->getCurrentElementCount() > 0;

        // k-NN strategy determined

        if (can_use_hnsw) {
            // ====== HNSW APPROXIMATE k-NN (FAST) ======
            // Using HNSW approximate k-NN

            // HNSW approximate k-NN (50-2000x faster)
            for (int i = 0; i < n_obs; i++) {
                try {
                    const float* query_point = &data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)];

                    // Search for k+1 neighbors (includes self)
                    auto result = model->ann_index->searchKnn(query_point, n_neighbors + 1);

                    // Extract neighbors, skipping self
                    std::vector<std::pair<float, int>> neighbors;
                    while (!result.empty()) {
                        auto pair = result.top();
                        result.pop();
                        int neighbor_id = static_cast<int>(pair.second);
                        if (neighbor_id != i) { // Skip self
                            neighbors.push_back({ pair.first, neighbor_id });
                        }
                    }

                    // Sort by distance and take k nearest
                    std::sort(neighbors.begin(), neighbors.end());
                    int actual_neighbors = std::min(static_cast<int>(neighbors.size()), n_neighbors);

                    for (int k = 0; k < actual_neighbors; k++) {
                        nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = neighbors[k].second;

                        // Convert HNSW distance to actual distance based on metric
                        float distance = neighbors[k].first;
                        switch (metric) {
                        case UWOT_METRIC_EUCLIDEAN:
                            distance = std::sqrt(std::max(0.0f, distance)); // L2Space returns squared distance
                            break;
                        case UWOT_METRIC_COSINE:
                            // InnerProductSpace returns -inner_product for unit vectors
                            // Convert to cosine distance: distance = 1 - similarity
                            distance = std::max(0.0f, std::min(2.0f, 1.0f + distance));
                            break;
                        case UWOT_METRIC_MANHATTAN:
                            // L1Space returns direct Manhattan distance
                            distance = std::max(0.0f, distance);
                            break;
                        default:
                            distance = std::max(0.0f, distance);
                            break;
                        }

                        nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] =
                            static_cast<double>(distance);
                    }

                    // Fill remaining slots if needed
                    for (int k = actual_neighbors; k < n_neighbors; k++) {
                        nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = 0;
                        nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = 1000.0;
                    }
                }
                catch (...) {
                    // Fallback for any HNSW errors
                    for (int k = 0; k < n_neighbors; k++) {
                        nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = (i + k + 1) % n_obs;
                        nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = 1000.0;
                    }
                }

                // Progress reporting every 10%
                if (progress_callback && i % (n_obs / 10 + 1) == 0) {
                    float percent = static_cast<float>(i) * 100.0f / static_cast<float>(n_obs);
                    auto elapsed = std::chrono::steady_clock::now() - start_time;
                    auto elapsed_sec = std::chrono::duration<double>(elapsed).count();
                    double remaining_sec = (elapsed_sec / (i + 1)) * (n_obs - i - 1);

                    char message[256];
                    snprintf(message, sizeof(message), "HNSW approx k-NN: %.1f%% (est. remaining: %.1fs)",
                        percent, remaining_sec);
                    progress_callback("HNSW k-NN Graph", i, n_obs, percent, message);
                }
            }

            if (progress_callback) {
                auto total_elapsed = std::chrono::steady_clock::now() - start_time;
                auto total_sec = std::chrono::duration<double>(total_elapsed).count();
                char final_message[256];
                snprintf(final_message, sizeof(final_message), "HNSW k-NN completed in %.2fs (approx mode)", total_sec);
                progress_callback("HNSW k-NN Graph", n_obs, n_obs, 100.0f, final_message);
            }

        }
        else {
            // ====== BRUTE-FORCE EXACT k-NN (SLOW BUT EXACT) ======
            // Using exact brute-force k-NN

            // Issue warnings for large datasets
            if (progress_callback) {
                const char* reason = force_exact_knn ? "exact k-NN forced" :
                    (!model || !model->space_factory) ? "HNSW not available" :
                    !model->space_factory->can_use_hnsw() ? "unsupported metric for HNSW" : "HNSW index missing";

                if (n_obs > 10000 || (static_cast<long long>(n_obs) * n_obs * n_dim) > 1e8) {
                    // Estimate time for large datasets
                    double est_operations = static_cast<double>(n_obs) * n_obs * n_dim;
                    double est_seconds = est_operations * 1e-9; // Conservative estimate: 1B ops/sec

                    char warning[512];
                    snprintf(warning, sizeof(warning),
                        "WARNING: Exact k-NN on %dx%d dataset (%s). Est. time: %.1f minutes. "
                        "Consider Euclidean/Cosine/Manhattan metrics for HNSW speedup.",
                        n_obs, n_dim, reason, est_seconds / 60.0);
                    progress_callback("Exact k-NN Graph", 0, n_obs, 0.0f, warning);
                }
                else {
                    char info[256];
                    snprintf(info, sizeof(info), "Exact k-NN mode (%s)", reason);
                    progress_callback("Exact k-NN Graph", 0, n_obs, 0.0f, info);
                }
            }

            // Original brute-force implementation with progress reporting
            for (int i = 0; i < n_obs; i++) {
                std::vector<std::pair<double, int>> distances;

                for (int j = 0; j < n_obs; j++) {
                    if (i == j) continue;

                    float dist = distance_metrics::compute_distance(
                        &data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],
                        &data[static_cast<size_t>(j) * static_cast<size_t>(n_dim)],
                        n_dim, metric);
                    distances.push_back({ static_cast<double>(dist), j });
                }

                std::partial_sort(distances.begin(),
                    distances.begin() + n_neighbors,
                    distances.end());

                for (int k = 0; k < n_neighbors; k++) {
                    nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = distances[static_cast<size_t>(k)].second;
                    nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = distances[static_cast<size_t>(k)].first;
                }

                // Progress reporting every 5%
                if (progress_callback && i % (n_obs / 20 + 1) == 0) {
                    float percent = static_cast<float>(i) * 100.0f / static_cast<float>(n_obs);
                    auto elapsed = std::chrono::steady_clock::now() - start_time;
                    auto elapsed_sec = std::chrono::duration<double>(elapsed).count();
                    double remaining_sec = (elapsed_sec / (i + 1)) * (n_obs - i - 1);

                    char message[256];
                    snprintf(message, sizeof(message), "Exact k-NN: %.1f%% (est. remaining: %.1fs)",
                        percent, remaining_sec);
                    progress_callback("Exact k-NN Graph", i, n_obs, percent, message);
                }
            }

            if (progress_callback) {
                auto total_elapsed = std::chrono::steady_clock::now() - start_time;
                auto total_sec = std::chrono::duration<double>(total_elapsed).count();
                char final_message[256];
                snprintf(final_message, sizeof(final_message), "Exact k-NN completed in %.2fs", total_sec);
                progress_callback("Exact k-NN Graph", n_obs, n_obs, 100.0f, final_message);
            }
        }
    }

    // Convert uwot smooth k-NN output to edge list format
    void convert_to_edges(const std::vector<int>& nn_indices,
        const std::vector<double>& nn_weights,
        int n_obs, int n_neighbors,
        std::vector<unsigned int>& heads,
        std::vector<unsigned int>& tails,
        std::vector<double>& weights) {

        // Use map to store symmetric edges and combine weights
        std::map<std::pair<int, int>, double> edge_map;

        for (int i = 0; i < n_obs; i++) {
            for (int k = 0; k < n_neighbors; k++) {
                int j = nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)];
                double weight = nn_weights[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)];

                // Add edge in both directions for symmetrization
                edge_map[{i, j}] += weight;
                edge_map[{j, i}] += weight;
            }
        }

        // Convert to edge list, avoiding duplicates
        for (const auto& edge : edge_map) {
            int i = edge.first.first;
            int j = edge.first.second;

            if (i < j) { // Only add each edge once
                heads.push_back(static_cast<unsigned int>(i));
                tails.push_back(static_cast<unsigned int>(j));
                weights.push_back(edge.second / 2.0); // Average the weights
            }
        }
    }

    // Calculate UMAP a,b parameters from spread and min_dist
    // Based on the official UMAP implementation curve fitting
    void calculate_ab_from_spread_and_min_dist(UwotModel* model) {
        float spread = model->spread;
        float min_dist = model->min_dist;

        // Handle edge cases
        if (spread <= 0.0f) spread = 1.0f;
        if (min_dist < 0.0f) min_dist = 0.0f;
        if (min_dist >= spread) {
            // If min_dist >= spread, use default values
            model->a = 1.929f;
            model->b = 0.7915f;
            return;
        }

        // Simplified curve fitting (C++ approximation of scipy.optimize.curve_fit)
        // Target: fit 1/(1 + a*x^(2*b)) to exponential decay
        // y = 1.0 for x < min_dist
        // y = exp(-(x - min_dist) / spread) for x >= min_dist

        // Key points for fitting:
        // At x = min_dist: y should be ~1.0
        // At x = spread: y should be ~exp(-1) ≈ 0.368
        // At x = 2*spread: y should be ~exp(-2) ≈ 0.135

        float x1 = min_dist + 0.001f; // Just above min_dist
        float x2 = spread;
        float y1 = 0.99f;  // Target at min_dist
        float y2 = std::exp(-1.0f); // Target at spread ≈ 0.368

        // Solve for b first using the ratio of the two points
        // 1/(1 + a*x1^(2b)) = y1 and 1/(1 + a*x2^(2b)) = y2
        // This gives us: (1/y1 - 1) / (1/y2 - 1) = (x1/x2)^(2b)
        float ratio_left = (1.0f / y1 - 1.0f) / (1.0f / y2 - 1.0f);
        float ratio_right = x1 / x2;

        if (ratio_left > 0 && ratio_right > 0) {
            model->b = std::log(ratio_left) / (2.0f * std::log(ratio_right));
            // Clamp b to reasonable range
            model->b = std::max(0.1f, std::min(model->b, 2.0f));

            // Now solve for a using the first point
            model->a = (1.0f / y1 - 1.0f) / std::pow(x1, 2.0f * model->b);
            // Clamp a to reasonable range
            model->a = std::max(0.001f, std::min(model->a, 1000.0f));
        }
        else {
            // Fallback to approximation based on spread/min_dist ratio
            float ratio = spread / (min_dist + 0.001f);
            model->b = std::max(0.5f, std::min(2.0f, std::log(ratio) / 2.0f));
            model->a = std::max(0.1f, std::min(10.0f, 1.0f / ratio));
        }
    }

    // Main fit function with progress reporting
    int uwot_fit_with_progress(UwotModel* model,
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
        int ef_search) {

        // Training function called

        if (!model || !data || !embedding || n_obs <= 0 || n_dim <= 0 ||
            embedding_dim <= 0 || n_neighbors <= 0 || n_epochs <= 0) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        if (embedding_dim > 50) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            model->n_vertices = n_obs;
            model->n_dim = n_dim;
            model->embedding_dim = embedding_dim;
            model->n_neighbors = n_neighbors;
            model->min_dist = min_dist;
            model->spread = spread;
            model->metric = metric;
            model->force_exact_knn = (force_exact_knn != 0); // Convert int to bool
            model->use_quantization = false; // PQ removed

            // Auto-scale HNSW parameters based on dataset size (if not explicitly set)
            if (M == -1) {  // Auto-scale flag
                if (n_obs < 50000) {
                    model->hnsw_M = 16;
                    model->hnsw_ef_construction = 64;
                    model->hnsw_ef_search = 32;
                }
                else if (n_obs < 1000000) {
                    model->hnsw_M = 32;
                    model->hnsw_ef_construction = 128;
                    model->hnsw_ef_search = 64;
                }
                else {
                    model->hnsw_M = 64;
                    model->hnsw_ef_construction = 128;
                    model->hnsw_ef_search = 128;
                }
            }
            else {
                // Use explicitly provided parameters
                model->hnsw_M = M;
                model->hnsw_ef_construction = ef_construction;
                model->hnsw_ef_search = ef_search;
            }

            // Suggestion 4: Auto-scale ef_search based on dim/size
            model->hnsw_ef_search = std::max(model->hnsw_ef_search, static_cast<int>(model->n_neighbors * std::log(static_cast<float>(n_obs)) / std::log(2.0f)));
            model->hnsw_ef_search = std::max(model->hnsw_ef_search, static_cast<int>(std::sqrt(static_cast<float>(n_dim)) * 2));  // Scale with sqrt(dim) for FP robustness

            // UNIFIED DATA PIPELINE from errors4.txt Solution 2
            // Use the SAME data for both HNSW index and k-NN computation
            std::vector<float> input_data(data, data + (static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim)));

            // Compute normalization parameters
            compute_normalization(input_data, n_obs, n_dim, model->feature_means, model->feature_stds);
            model->use_normalization = true;

            // Determine normalization mode and apply consistently
            auto norm_mode = hnsw_utils::NormalizationPipeline::determine_normalization_mode(metric);
            model->normalization_mode = norm_mode;

            // Apply consistent normalization to create SINGLE unified dataset
            std::vector<float> normalized_data;
            hnsw_utils::NormalizationPipeline::normalize_data_consistent(
                input_data, normalized_data, n_obs, n_dim,
                model->feature_means, model->feature_stds, norm_mode);

            // Fix 1: Ensure unit normalization for cosine distance
            if (metric == UWOT_METRIC_COSINE) {
                // Ensure each vector is unit-normalized (HNSW InnerProductSpace expects normalized vectors for cosine)
                for (int i = 0; i < n_obs; ++i) {
                    float norm = 0.0f;
                    for (int j = 0; j < n_dim; ++j) {
                        float v = normalized_data[static_cast<size_t>(i) * n_dim + j];
                        norm += v * v;
                    }
                    norm = std::sqrt(std::max(1e-12f, norm));
                    for (int j = 0; j < n_dim; ++j) {
                        normalized_data[static_cast<size_t>(i) * n_dim + j] /= norm;
                    }
                }
            }

            if (progress_callback) {
                progress_callback(10, 100, 10.0f);  // Data normalization complete
            }

            // CRITICAL FIX: Create HNSW index BEFORE k-NN graph so build_knn_graph can use it
            if (!model->space_factory->create_space(metric, n_dim)) {
                return UWOT_ERROR_MEMORY;
            }

            // Memory estimation for HNSW index - calculate expected memory usage
            size_t estimated_memory_mb = ((size_t)n_obs * model->hnsw_M * 4 * 2) / (1024 * 1024);
            // Creating HNSW index with calculated parameters

            model->ann_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                model->space_factory->get_space(), n_obs, model->hnsw_M, model->hnsw_ef_construction);
            model->ann_index->setEf(model->hnsw_ef_search);  // Set query-time ef parameter

            // Add all points to HNSW index using the SAME normalized data
            for (int i = 0; i < n_obs; i++) {
                model->ann_index->addPoint(
                    &normalized_data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],
                    static_cast<hnswlib::labeltype>(i));
            }

            // HNSW index construction completed

            // Use same data for BOTH HNSW and exact k-NN - this is the key fix!

            // Compute comprehensive neighbor statistics on the SAME data as HNSW
            compute_neighbor_statistics(model, normalized_data);

            // Build k-NN graph using SAME prepared data as HNSW index - INDEX NOW AVAILABLE!
            std::vector<int> nn_indices;
            std::vector<double> nn_distances;

            // Create wrapper for passing warnings to v2 callback if available
            uwot_progress_callback_v2 wrapped_callback = nullptr;
            if (g_v2_callback) {
                wrapped_callback = g_v2_callback;  // Pass warnings directly to v2 callback
            }

            build_knn_graph(normalized_data, n_obs, n_dim, n_neighbors, metric, model,
                nn_indices, nn_distances, force_exact_knn, wrapped_callback);

            // Use uwot smooth_knn to compute weights
            std::vector<std::size_t> nn_ptr = { static_cast<std::size_t>(n_neighbors) };
            std::vector<double> target = { std::log2(static_cast<double>(n_neighbors)) };
            std::vector<double> nn_weights(nn_indices.size());
            std::vector<double> sigmas, rhos;
            std::atomic<std::size_t> n_search_fails{ 0 };

            uwot::smooth_knn(0, static_cast<std::size_t>(n_obs), nn_distances, nn_ptr, false, target,
                1.0, 1e-5, 64, 0.001,
                uwot::mean_average(nn_distances), false,
                nn_weights, sigmas, rhos, n_search_fails);

            // Fix 2: Apply minimal weight floor to preserve relative differences
            const double MIN_WEIGHT = 1e-6;
            for (size_t wi = 0; wi < nn_weights.size(); ++wi) {
                if (nn_weights[wi] < MIN_WEIGHT) nn_weights[wi] = MIN_WEIGHT;
            }

            // Convert to edge format for optimization
            convert_to_edges(nn_indices, nn_weights, n_obs, n_neighbors,
                model->positive_head, model->positive_tail, model->positive_weights);

            // Store k-NN data for transform (flattened format)
            model->nn_indices = nn_indices;
            model->nn_distances.resize(nn_distances.size());
            model->nn_weights.resize(nn_weights.size());
            for (size_t i = 0; i < nn_distances.size(); i++) {
                model->nn_distances[i] = static_cast<float>(nn_distances[i]);
                // Fix 2: Ensure no overly-large floor when converting to float
                model->nn_weights[i] = static_cast<float>(std::max<double>(nn_weights[i], 1e-6));
            }

            // Initialize embedding
            model->embedding.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim));

            // Thread-safe random initialization
#pragma omp parallel if(n_obs > 1000)
            {
                // Each thread gets its own generator to avoid race conditions
                thread_local std::mt19937 gen(42 + omp_get_thread_num());
                thread_local std::normal_distribution<float> dist(0.0f, 1e-4f);

#pragma omp for
                for (int i = 0; i < static_cast<int>(static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim)); i++) {
                    model->embedding[i] = dist(gen);
                }
            }

            // Calculate UMAP parameters from spread and min_dist
            calculate_ab_from_spread_and_min_dist(model);

            // Direct UMAP optimization implementation with progress reporting
            const float learning_rate = 1.0f;
            std::mt19937 rng(42);
            std::uniform_int_distribution<size_t> vertex_dist(0, static_cast<size_t>(n_obs) - 1);

            // Enhanced progress reporting setup
            int progress_interval = std::max(1, n_epochs / 100);  // Report every 1% progress
            auto last_report_time = std::chrono::steady_clock::now();

            // Only show console output if no callback provided
            if (!progress_callback) {
                std::printf("UMAP Training Progress:\n");
                std::printf("[                    ] 0%% (Epoch 0/%d)\n", n_epochs);
                std::fflush(stdout);
            }

            for (int epoch = 0; epoch < n_epochs; epoch++) {
                float alpha = learning_rate * (1.0f - static_cast<float>(epoch) / static_cast<float>(n_epochs));

                // Loss calculation for progress reporting
                float epoch_loss = 0.0f;
                int loss_samples = 0;

                // Process positive edges (attractive forces)
                for (size_t edge_idx = 0; edge_idx < model->positive_head.size(); edge_idx++) {
                    size_t i = static_cast<size_t>(model->positive_head[edge_idx]);
                    size_t j = static_cast<size_t>(model->positive_tail[edge_idx]);

                    // Compute squared distance
                    float dist_sq = 0.0f;
                    for (int d = 0; d < embedding_dim; d++) {
                        float diff = model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -
                            model->embedding[j * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];
                        dist_sq += diff * diff;
                    }

                    if (dist_sq > std::numeric_limits<float>::epsilon()) {
                        // UMAP attractive gradient: -2*2ab * d^(2b-2) / (1 + a*d^(2b))
                        float pd2b = std::pow(dist_sq, model->b);
                        float grad_coeff = (-2.0f * model->a * model->b * pd2b) /
                            (dist_sq * (model->a * pd2b + 1.0f));

                        // Apply clamping
                        grad_coeff = std::max(-4.0f, std::min(4.0f, grad_coeff));

                        for (int d = 0; d < embedding_dim; d++) {
                            float diff = model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -
                                model->embedding[j * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];
                            float grad = alpha * grad_coeff * diff;
                            model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] += grad;
                            model->embedding[j * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -= grad;
                        }

                        // Accumulate attractive force loss (UMAP cross-entropy: attractive term)
                        if (loss_samples < 1000) { // Sample subset for performance
                            float attractive_prob = 1.0f / (1.0f + model->a * pd2b);
                            epoch_loss += -std::log(attractive_prob + 1e-8f);  // -log(P_attract)
                            loss_samples++;
                        }
                    }

                    // Negative sampling (5 samples per positive edge)
                    for (int neg = 0; neg < 5; neg++) {
                        size_t k = vertex_dist(rng);
                        if (k == i || k == j) continue;

                        float neg_dist_sq = 0.0f;
                        for (int d = 0; d < embedding_dim; d++) {
                            float diff = model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -
                                model->embedding[k * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];
                            neg_dist_sq += diff * diff;
                        }

                        if (neg_dist_sq > std::numeric_limits<float>::epsilon()) {
                            // UMAP repulsive gradient: 2b / ((0.001 + d^2) * (1 + a*d^(2b)))
                            float pd2b = std::pow(neg_dist_sq, model->b);
                            float grad_coeff = (2.0f * model->b) /
                                ((0.001f + neg_dist_sq) * (model->a * pd2b + 1.0f));

                            // Apply clamping
                            grad_coeff = std::max(-4.0f, std::min(4.0f, grad_coeff));

                            for (int d = 0; d < embedding_dim; d++) {
                                float diff = model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -
                                    model->embedding[k * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];
                                float grad = alpha * grad_coeff * diff;
                                model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] += grad;
                            }

                            // Accumulate repulsive force loss (UMAP cross-entropy: repulsive term)
                            if (loss_samples < 1000) { // Sample subset for performance
                                float repulsive_prob = 1.0f / (1.0f + model->a * std::pow(neg_dist_sq, model->b));
                                epoch_loss += -std::log(1.0f - repulsive_prob + 1e-8f);  // -log(1 - P_repulse)
                                loss_samples++;
                            }
                        }
                    }
                }

                // Adaptive progress reporting: more frequent for early epochs
                bool should_report = (epoch < 10) ||                        // Report first 10 epochs
                    (epoch % progress_interval == 0) ||       // Regular interval
                    (epoch == n_epochs - 1);                  // Final epoch

                if (should_report) {
                    float percent = (static_cast<float>(epoch + 1) / static_cast<float>(n_epochs)) * 100.0f;

                    // Calculate average loss for this epoch (shared by both callback and console)
                    float avg_loss = loss_samples > 0 ? epoch_loss / loss_samples : 0.0f;

                    if (progress_callback) {
                        // Use callback for C# integration - pass loss info in global variable
                        g_current_epoch_loss = avg_loss;  // Store for v2 callback wrapper
                        progress_callback(epoch + 1, n_epochs, percent);
                    }
                    else {
                        // Console output for C++ testing
                        int percent_int = static_cast<int>(percent);
                        int filled = percent_int / 5;  // 20 characters for 100%

                        auto current_time = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_report_time);

                        std::printf("\r[");
                        for (int i = 0; i < 20; i++) {
                            if (i < filled) std::printf("=");
                            else if (i == filled && percent_int % 5 >= 2) std::printf(">");
                            else std::printf(" ");
                        }

                        std::printf("] %d%% (Epoch %d/%d) Loss: %.3f [%lldms]",
                            percent_int, epoch + 1, n_epochs, avg_loss, static_cast<long long>(elapsed.count()));
                        std::fflush(stdout);

                        last_report_time = current_time;
                    }
                }
            }

            if (!progress_callback) {
                std::printf("\nTraining completed!\n");
                std::fflush(stdout);
            }

            // Fix 6: Bounds-checked element-wise copy instead of unsafe memcpy
            size_t expected = static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim);
            if (model->embedding.size() < expected) {
                return UWOT_ERROR_MEMORY;
            }
            for (size_t i = 0; i < expected; ++i) {
                embedding[i] = model->embedding[i];
            }

            model->is_fitted = true;
            return UWOT_SUCCESS;

        }
        catch (...) {
            return UWOT_ERROR_MEMORY;
        }
    }

    // Enhanced v2 function with loss reporting - delegates to existing function
    int uwot_fit_with_progress_v2(UwotModel* model,
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
        int ef_search) {

        // Enhanced training function with v2 progress callbacks

        if (!model || !data || !embedding || n_obs <= 0 || n_dim <= 0 ||
            embedding_dim <= 0 || n_neighbors <= 0 || n_epochs <= 0) {
            if (progress_callback) {
                progress_callback("Error", 0, 1, 0.0f, "Invalid parameters: model, data, or embedding parameters are invalid");
            }
            return UWOT_ERROR_INVALID_PARAMS;
        }

        if (embedding_dim > 50) {
            if (progress_callback) {
                progress_callback("Error", 0, 1, 0.0f, "Invalid parameter: embedding dimension must be <= 50");
            }
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            // Create v1 callback wrapper for epoch progress (with loss)
            // The global callback is managed separately via SetGlobalCallback API
            static thread_local uwot_progress_callback_v2 g_local_v2_callback = nullptr;
            g_local_v2_callback = progress_callback;

            uwot_progress_callback v1_callback = nullptr;
            if (progress_callback) {
                v1_callback = [](int epoch, int total_epochs, float percent) {
                    if (g_local_v2_callback) {
                        // Format loss message with current loss value
                        char message[256];
                        snprintf(message, sizeof(message), "Loss: %.3f", g_current_epoch_loss);
                        g_local_v2_callback("Training", epoch, total_epochs, percent, message);
                    }
                    };
            }

            return fit_utils::uwot_fit_with_progress(model, data, n_obs, n_dim, embedding_dim,
                n_neighbors, min_dist, spread, n_epochs, metric, embedding,
                v1_callback, force_exact_knn, M, ef_construction, ef_search);

        }
        catch (...) {
            const char* error_msg = "An error occurred during training";
            if (progress_callback) {
                progress_callback("Error", 0, 1, 0.0f, error_msg);
            }
            else {
                send_error_to_callback(error_msg);
            }
            return UWOT_ERROR_MEMORY;
        }
    }

}