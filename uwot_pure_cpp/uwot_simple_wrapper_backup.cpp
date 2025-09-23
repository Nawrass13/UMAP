#include "uwot_simple_wrapper.h"
#include "uwot_progress_utils.h"
#include "uwot_hnsw_utils.h"
#include "uwot_model.h"
#include "uwot_persistence.h"
#include "uwot_fit.h"
#include "uwot_transform.h"
#include "uwot_quantization.h"
#include "smooth_knn.h"
#include "optimize.h"
#include "gradient.h"
#include "transform.h"
#include "update.h"
#include "epoch.h"

// HNSW library integration
#include "hnswlib.h"
#include "hnswalg.h"
#include "space_l2.h"
#include "space_ip.h"

#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <memory>

// LZ4 compression support for fast model storage
#include "lz4.h"
#define COMPRESSION_AVAILABLE true
#include <random>
#include <algorithm>
#include <cstring>
#include <map>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>










// Enhanced HNSW stream operations with improved temporary file management
namespace hnsw_stream_utils {

    // Generate unique temporary filename based on thread ID and timestamp
    std::string generate_unique_temp_filename(const std::string& base_name) {
        auto now = std::chrono::steady_clock::now();
        auto timestamp = now.time_since_epoch().count();
        std::ostringstream oss;
        oss << base_name << "_" << std::this_thread::get_id() << "_" << timestamp << ".tmp";
        return oss.str();
    }

    // Save HNSW index to stream with improved temporary file handling
    void save_hnsw_to_stream(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
        // Generate unique temporary filename to avoid conflicts
        std::string temp_filename = generate_unique_temp_filename("hnsw_save");

        try {
            // Save HNSW index to temporary file
            hnsw_index->saveIndex(temp_filename);

            // Read the temporary file and stream it directly
            std::ifstream temp_file(temp_filename, std::ios::binary);
            if (temp_file.is_open()) {
                // Use efficient stream buffer copying
                output << temp_file.rdbuf();
                temp_file.close();
            }
            else {
                throw std::runtime_error("Failed to open temporary HNSW file for reading");
            }

            // Clean up temporary file immediately
            temp_utils::safe_remove_file(temp_filename);

        }
        catch (...) {
            // Ensure cleanup even on exception
            temp_utils::safe_remove_file(temp_filename);
            throw;
        }
    }

    // Load HNSW index from stream with improved temporary file handling
    void load_hnsw_from_stream(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
        hnswlib::SpaceInterface<float>* space, size_t hnsw_size) {
        // Generate unique temporary filename to avoid conflicts
        std::string temp_filename = generate_unique_temp_filename("hnsw_load");

        try {
            // Write the stream data to temporary file
            std::ofstream temp_file(temp_filename, std::ios::binary);
            if (temp_file.is_open()) {
                // Efficient stream copying with proper size handling
                std::vector<char> buffer(8192); // Larger buffer for efficiency
                size_t remaining = hnsw_size;

                while (remaining > 0 && input.good()) {
                    size_t to_read = std::min(buffer.size(), remaining);
                    input.read(buffer.data(), static_cast<std::streamsize>(to_read));
                    std::streamsize actually_read = input.gcount();

                    if (actually_read > 0) {
                        temp_file.write(buffer.data(), actually_read);
                        remaining -= static_cast<size_t>(actually_read);
                    }
                }
                temp_file.close();

                // Load from temporary file using HNSW API
                hnsw_index->loadIndex(temp_filename, space);

            }
            else {
                throw std::runtime_error("Failed to create temporary HNSW file for writing");
            }

            // Clean up temporary file immediately
            temp_utils::safe_remove_file(temp_filename);

        }
        catch (...) {
            // Ensure cleanup even on exception
            temp_utils::safe_remove_file(temp_filename);
            throw;
        }
    }
}

// Stream wrapper functions
void save_hnsw_to_stream(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
    hnsw_stream_utils::save_hnsw_to_stream(output, hnsw_index);
}

void load_hnsw_from_stream(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
    hnswlib::SpaceInterface<float>* space, size_t hnsw_size) {
    hnsw_stream_utils::load_hnsw_from_stream(input, hnsw_index, space, hnsw_size);
}


void load_hnsw_from_stream_compressed(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
    hnswlib::SpaceInterface<float>* space) {
    std::string temp_filename;

    try {
        // Read LZ4 compression headers with validation
        size_t uncompressed_size;
        int compressed_size;

        input.read(reinterpret_cast<char*>(&uncompressed_size), sizeof(size_t));
        if (!input.good()) {
            throw std::runtime_error("Failed to read uncompressed size header");
        }

        input.read(reinterpret_cast<char*>(&compressed_size), sizeof(int));
        if (!input.good()) {
            throw std::runtime_error("Failed to read compressed size header");
        }

        // Sanity checks to prevent crashes from corrupted data
        const size_t max_uncompressed = 1ULL * 1024 * 1024 * 1024; // 1GB limit
        const size_t max_compressed = 500 * 1024 * 1024; // 500MB limit

        if (uncompressed_size == 0 || uncompressed_size > max_uncompressed) {
            throw std::runtime_error("Invalid uncompressed size in HNSW data");
        }

        if (compressed_size <= 0 || static_cast<size_t>(compressed_size) > max_compressed) {
            throw std::runtime_error("Invalid compressed size in HNSW data");
        }

        // Read compressed data with validation
        std::vector<char> compressed_data;
        try {
            compressed_data.resize(compressed_size);
        }
        catch (const std::bad_alloc&) {
            throw std::runtime_error("Failed to allocate memory for compressed HNSW data");
        }

        input.read(compressed_data.data(), compressed_size);
        if (!input.good() || input.gcount() != compressed_size) {
            throw std::runtime_error("Failed to read compressed HNSW data from stream");
        }

        // Decompress data using LZ4 with validation
        std::vector<char> uncompressed_data;
        try {
            uncompressed_data.resize(uncompressed_size);
        }
        catch (const std::bad_alloc&) {
            throw std::runtime_error("Failed to allocate memory for uncompressed HNSW data");
        }

        int result = LZ4_decompress_safe(
            compressed_data.data(),
            uncompressed_data.data(),
            compressed_size,
            static_cast<int>(uncompressed_size));

        if (result != static_cast<int>(uncompressed_size)) {
            throw std::runtime_error("LZ4 HNSW decompression failed - corrupted data or invalid format");
        }

        // Write to temporary file for HNSW loading
        temp_filename = hnsw_stream_utils::generate_unique_temp_filename("hnsw_decompress");
        std::ofstream temp_file(temp_filename, std::ios::binary);
        if (!temp_file.is_open()) {
            throw std::runtime_error("Failed to create temporary HNSW file for decompression");
        }

        temp_file.write(uncompressed_data.data(), uncompressed_size);
        temp_file.close();

        if (!temp_file.good()) {
            throw std::runtime_error("Failed to write decompressed HNSW data to temporary file");
        }

        // Load from temporary file using HNSW API
        hnsw_index->loadIndex(temp_filename, space);

        // Clean up temporary file
        temp_utils::safe_remove_file(temp_filename);

    }
    catch (const std::exception& e) {
        // Clean up temporary file if it was created
        if (!temp_filename.empty()) {
            temp_utils::safe_remove_file(temp_filename);
        }
        throw std::runtime_error(std::string("HNSW decompression failed: ") + e.what());
    }
    catch (...) {
        // Clean up temporary file if it was created
        if (!temp_filename.empty()) {
            temp_utils::safe_remove_file(temp_filename);
        }
        throw std::runtime_error("HNSW decompression failed with unknown error");
    }
}

extern "C" {

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

    UWOT_API UwotModel* uwot_create() {
        return model_utils::create_model();
    }

    // OLD uwot_fit REMOVED - ALL FUNCTIONS NOW USE UNIFIED PIPELINE

    UWOT_API int uwot_fit_with_progress(UwotModel* model,
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

        printf("🔥 DEBUG: uwot_fit_with_progress CALLED (force_exact_knn=%d)\n", force_exact_knn);

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
            fit_utils::compute_normalization(input_data, n_obs, n_dim, model->feature_means, model->feature_stds);
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

            // Memory estimation for HNSW index
            size_t estimated_memory_mb = ((size_t)n_obs * model->hnsw_M * 4 * 2) / (1024 * 1024);
            printf("[HNSW] Creating index: %d points, estimated %zuMB memory, M=%d, ef_c=%d\n",
                n_obs, estimated_memory_mb, model->hnsw_M, model->hnsw_ef_construction);

            model->ann_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                model->space_factory->get_space(), n_obs, model->hnsw_M, model->hnsw_ef_construction);
            model->ann_index->setEf(model->hnsw_ef_search);  // Set query-time ef parameter

            // Add all points to HNSW index using the SAME normalized data
            for (int i = 0; i < n_obs; i++) {
                model->ann_index->addPoint(
                    &normalized_data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],
                    static_cast<hnswlib::labeltype>(i));
            }

            printf("[HNSW] Built index with %d points in %dD space\n", n_obs, n_dim);

            // Use same data for BOTH HNSW and exact k-NN - this is the key fix!

            // Compute comprehensive neighbor statistics on the SAME data as HNSW
            fit_utils::compute_neighbor_statistics(model, normalized_data);

            // Build k-NN graph using SAME prepared data as HNSW index - INDEX NOW AVAILABLE!
            std::vector<int> nn_indices;
            std::vector<double> nn_distances;

            // Create wrapper for passing warnings to v2 callback if available
            uwot_progress_callback_v2 wrapped_callback = nullptr;
            if (g_v2_callback) {
                wrapped_callback = g_v2_callback;  // Pass warnings directly to v2 callback
            }

            fit_utils::build_knn_graph(normalized_data, n_obs, n_dim, n_neighbors, metric, model,
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
            fit_utils::convert_to_edges(nn_indices, nn_weights, n_obs, n_neighbors,
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
                            float attractive_term = 1.0f / (1.0f + model->a * pd2b);
                            epoch_loss += -std::log(attractive_term + 1e-8f);  // Negative log-probability for attraction
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
                                float repulsive_term = model->a * std::pow(neg_dist_sq, model->b);
                                epoch_loss += std::log(1.0f + repulsive_term + 1e-8f);  // Log(1 + repulsive) for repulsion
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

    // OLD uwot_fit_with_enhanced_progress REMOVED - ALL FUNCTIONS NOW USE UNIFIED PIPELINE

    // Enhanced v2 function with loss reporting - delegates to existing function
    // The loss reporting will be added by modifying the core training loop
    UWOT_API int uwot_fit_with_progress_v2(UwotModel* model,
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

        printf("🔥 DEBUG: uwot_fit_with_progress_v2 CALLED (force_exact_knn=%d)\n", force_exact_knn);

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

            return uwot_fit_with_progress(model, data, n_obs, n_dim, embedding_dim,
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

    // Global callback management functions
    UWOT_API void uwot_set_global_callback(uwot_progress_callback_v2 callback) {
        g_v2_callback = callback;
    }

    UWOT_API void uwot_clear_global_callback() {
        g_v2_callback = nullptr;
    }

    UWOT_API int uwot_transform(UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding) {

        return transform_utils::uwot_transform(model, new_data, n_new_obs, n_dim, embedding);
    }

    UWOT_API int uwot_transform_detailed(
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
        return transform_utils::uwot_transform_detailed(model, new_data, n_new_obs, n_dim, embedding,
            nn_indices, nn_distances, confidence_score, outlier_level, percentile_rank, z_score);
    }

    UWOT_API int uwot_get_model_info(UwotModel* model,
        int* n_vertices,
        int* n_dim,
        int* embedding_dim,
        int* n_neighbors,
        float* min_dist,
        float* spread,
        UwotMetric* metric,
        int* hnsw_M,
        int* hnsw_ef_construction,
        int* hnsw_ef_search) {
        if (!model) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        if (n_vertices) *n_vertices = model->n_vertices;
        if (n_dim) *n_dim = model->n_dim;
        if (embedding_dim) *embedding_dim = model->embedding_dim;
        if (n_neighbors) *n_neighbors = model->n_neighbors;
        if (min_dist) *min_dist = model->min_dist;
        if (spread) *spread = model->spread;
        if (metric) *metric = model->metric;
        if (hnsw_M) *hnsw_M = model->hnsw_M;
        if (hnsw_ef_construction) *hnsw_ef_construction = model->hnsw_ef_construction;
        if (hnsw_ef_search) *hnsw_ef_search = model->hnsw_ef_search;

        return UWOT_SUCCESS;
    }

    UWOT_API int uwot_save_model(UwotModel* model, const char* filename) {
        return persistence_utils::save_model(model, filename);
    }

    UWOT_API UwotModel* uwot_load_model(const char* filename) {
        return persistence_utils::load_model(filename);
    }

    UWOT_API const char* uwot_get_error_message(int error_code) {
        return model_utils::get_error_message(error_code);
    }

    UWOT_API const char* uwot_get_metric_name(UwotMetric metric) {
        return model_utils::get_metric_name(metric);
    }

    UWOT_API int uwot_get_embedding_dim(UwotModel* model) {
        return model_utils::get_embedding_dim(model);
    }

    UWOT_API int uwot_get_n_vertices(UwotModel* model) {
        return model_utils::get_n_vertices(model);
    }

    UWOT_API int uwot_is_fitted(UwotModel* model) {
        return model_utils::is_fitted(model);
    }

    UWOT_API const char* uwot_get_version() {
        return model_utils::get_version();
    }

} // extern "C"
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
                    for (int d = 0; d < model->embedding_dim; d++) {
                        float coord = 0.0f;
                        for (size_t k = 0; k < nn_indices.size(); k++) {
                            coord += model->embedding[static_cast<size_t>(nn_indices[k]) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] * nn_weights[k];
                        }
                        new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] = coord;
                    }
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
    UWOT_API int uwot_transform_detailed(
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

                // Use stored normalization mode from training
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
    UWOT_API int uwot_get_model_info(UwotModel* model,
        int* n_vertices,
        int* n_dim,
        int* embedding_dim,
        int* n_neighbors,
        float* min_dist,
        float* spread,
        UwotMetric* metric,
        int* hnsw_M,
        int* hnsw_ef_construction,
        int* hnsw_ef_search) {
        if (!model) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        if (n_vertices) *n_vertices = model->n_vertices;
        if (n_dim) *n_dim = model->n_dim;
        if (embedding_dim) *embedding_dim = model->embedding_dim;
        if (n_neighbors) *n_neighbors = model->n_neighbors;
        if (min_dist) *min_dist = model->min_dist;
        if (spread) *spread = model->spread;
        if (metric) *metric = model->metric;
        if (hnsw_M) *hnsw_M = model->hnsw_M;
        if (hnsw_ef_construction) *hnsw_ef_construction = model->hnsw_ef_construction;
        if (hnsw_ef_search) *hnsw_ef_search = model->hnsw_ef_search;

        return UWOT_SUCCESS;
    }

    UWOT_API int uwot_save_model(UwotModel* model, const char* filename) {
        return persistence_utils::save_model(model, filename);
    }

    UWOT_API UwotModel* uwot_load_model(const char* filename) {
        return persistence_utils::load_model(filename);
    }

    UWOT_API void uwot_destroy(UwotModel* model) {
        model_utils::destroy_model(model);
    }

    UWOT_API const char* uwot_get_error_message(int error_code) {
        return model_utils::get_error_message(error_code);
    }

    UWOT_API const char* uwot_get_metric_name(UwotMetric metric) {
        return model_utils::get_metric_name(metric);
    }

    UWOT_API int uwot_get_embedding_dim(UwotModel* model) {
        return model_utils::get_embedding_dim(model);
    }

    UWOT_API int uwot_get_n_vertices(UwotModel* model) {
        return model_utils::get_n_vertices(model);
    }

    UWOT_API int uwot_is_fitted(UwotModel* model) {
        return model_utils::is_fitted(model);
    }

    UWOT_API const char* uwot_get_version() {
        return model_utils::get_version();
    }

} // extern "C"