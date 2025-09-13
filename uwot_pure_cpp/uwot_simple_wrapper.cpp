#include "uwot_simple_wrapper.h"
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
#include <vector>
#include <memory>
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
#include <filesystem>

// Cross-platform file utilities
namespace temp_utils {
    bool safe_remove_file(const std::string& filename) {
        try {
            return std::filesystem::remove(filename);
        } catch (...) {
            // Fallback to C remove if filesystem fails
            return std::remove(filename.c_str()) == 0;
        }
    }
}

struct UwotModel {
    // Model parameters
    int n_vertices;
    int n_dim;
    int embedding_dim;
    int n_neighbors;
    float min_dist;
    UwotMetric metric;
    float a, b; // UMAP curve parameters
    bool is_fitted;

    // HNSW Index for fast neighbor search (replaces training_data storage)
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> ann_index;
    std::unique_ptr<hnswlib::L2Space> l2_space;
    std::unique_ptr<hnswlib::InnerProductSpace> ip_space;

    // Normalization parameters (moved from C#)
    std::vector<float> feature_means;
    std::vector<float> feature_stds;
    bool use_normalization;

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

    UwotModel() : n_vertices(0), n_dim(0), embedding_dim(2), n_neighbors(15),
        min_dist(0.1f), metric(UWOT_METRIC_EUCLIDEAN), a(1.929f), b(0.7915f),
        is_fitted(false), use_normalization(false),
        min_neighbor_distance(0.0f), mean_neighbor_distance(0.0f),
        std_neighbor_distance(0.0f), p95_neighbor_distance(0.0f),
        p99_neighbor_distance(0.0f), mild_outlier_threshold(0.0f),
        extreme_outlier_threshold(0.0f) {
    }
};

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

// Helper function to apply normalization
void normalize_data(const std::vector<float>& input_data, std::vector<float>& output_data,
    int n_obs, int n_dim, const std::vector<float>& means, const std::vector<float>& stds) {
    output_data.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim));

    for (int i = 0; i < n_obs; i++) {
        for (int j = 0; j < n_dim; j++) {
            size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
            output_data[idx] = (input_data[idx] - means[j]) / stds[j];
        }
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
                    all_distances.push_back(std::sqrt(pair.first)); // HNSW returns squared distances
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

    // Outlier thresholds
    model->mild_outlier_threshold = model->mean_neighbor_distance + 2.5f * model->std_neighbor_distance;
    model->extreme_outlier_threshold = model->mean_neighbor_distance + 4.0f * model->std_neighbor_distance;

    printf("[STATS] Neighbor distances - min: %.4f, mean: %.4f � %.4f, p95: %.4f, p99: %.4f\n",
        model->min_neighbor_distance, model->mean_neighbor_distance,
        model->std_neighbor_distance, model->p95_neighbor_distance, model->p99_neighbor_distance);
    printf("[STATS] Outlier thresholds - mild: %.4f, extreme: %.4f\n",
        model->mild_outlier_threshold, model->extreme_outlier_threshold);
}

// Distance metric implementations
namespace distance_metrics {

    float euclidean_distance(const float* a, const float* b, int dim) {
        float dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }

    float cosine_distance(const float* a, const float* b, int dim) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

        for (int i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);

        if (norm_a < 1e-10f || norm_b < 1e-10f) return 1.0f;

        float cosine_sim = dot / (norm_a * norm_b);
        cosine_sim = std::max(-1.0f, std::min(1.0f, cosine_sim));

        return 1.0f - cosine_sim;
    }

    float manhattan_distance(const float* a, const float* b, int dim) {
        float dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            dist += std::abs(a[i] - b[i]);
        }
        return dist;
    }

    float correlation_distance(const float* a, const float* b, int dim) {
        float mean_a = 0.0f, mean_b = 0.0f;
        for (int i = 0; i < dim; ++i) {
            mean_a += a[i];
            mean_b += b[i];
        }
        mean_a /= static_cast<float>(dim);
        mean_b /= static_cast<float>(dim);

        float num = 0.0f, den_a = 0.0f, den_b = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff_a = a[i] - mean_a;
            float diff_b = b[i] - mean_b;
            num += diff_a * diff_b;
            den_a += diff_a * diff_a;
            den_b += diff_b * diff_b;
        }

        if (den_a < 1e-10f || den_b < 1e-10f) return 1.0f;

        float correlation = num / std::sqrt(den_a * den_b);
        correlation = std::max(-1.0f, std::min(1.0f, correlation));

        return 1.0f - correlation;
    }

    float hamming_distance(const float* a, const float* b, int dim) {
        int different = 0;
        for (int i = 0; i < dim; ++i) {
            if (std::abs(a[i] - b[i]) > 1e-6f) {
                different++;
            }
        }
        return static_cast<float>(different) / static_cast<float>(dim);
    }

    float compute_distance(const float* a, const float* b, int dim, UwotMetric metric) {
        switch (metric) {
        case UWOT_METRIC_EUCLIDEAN:
            return euclidean_distance(a, b, dim);
        case UWOT_METRIC_COSINE:
            return cosine_distance(a, b, dim);
        case UWOT_METRIC_MANHATTAN:
            return manhattan_distance(a, b, dim);
        case UWOT_METRIC_CORRELATION:
            return correlation_distance(a, b, dim);
        case UWOT_METRIC_HAMMING:
            return hamming_distance(a, b, dim);
        default:
            return euclidean_distance(a, b, dim);
        }
    }
}

// Build k-NN graph using specified distance metric
void build_knn_graph(const std::vector<float>& data, int n_obs, int n_dim,
    int n_neighbors, UwotMetric metric,
    std::vector<int>& nn_indices, std::vector<double>& nn_distances) {

    nn_indices.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));
    nn_distances.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));

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
    void save_hnsw_to_stream(std::ostream &output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
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
            } else {
                throw std::runtime_error("Failed to open temporary HNSW file for reading");
            }

            // Clean up temporary file immediately
            temp_utils::safe_remove_file(temp_filename);

        } catch (...) {
            // Ensure cleanup even on exception
            temp_utils::safe_remove_file(temp_filename);
            throw;
        }
    }

    // Load HNSW index from stream with improved temporary file handling
    void load_hnsw_from_stream(std::istream &input, hnswlib::HierarchicalNSW<float>* hnsw_index,
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

            } else {
                throw std::runtime_error("Failed to create temporary HNSW file for writing");
            }

            // Clean up temporary file immediately
            temp_utils::safe_remove_file(temp_filename);

        } catch (...) {
            // Ensure cleanup even on exception
            temp_utils::safe_remove_file(temp_filename);
            throw;
        }
    }
}

// Stream wrapper functions
void save_hnsw_to_stream(std::ostream &output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
    hnsw_stream_utils::save_hnsw_to_stream(output, hnsw_index);
}

void load_hnsw_from_stream(std::istream &input, hnswlib::HierarchicalNSW<float>* hnsw_index,
                          hnswlib::SpaceInterface<float>* space, size_t hnsw_size) {
    hnsw_stream_utils::load_hnsw_from_stream(input, hnsw_index, space, hnsw_size);
}

extern "C" {

    UWOT_API UwotModel* uwot_create() {
        try {
            return new UwotModel();
        }
        catch (...) {
            return nullptr;
        }
    }

    UWOT_API int uwot_fit(UwotModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float min_dist,
        int n_epochs,
        UwotMetric metric,
        float* embedding) {

        return uwot_fit_with_progress(model, data, n_obs, n_dim, embedding_dim,
            n_neighbors, min_dist, n_epochs, metric,
            embedding, nullptr);
    }

    UWOT_API int uwot_fit_with_progress(UwotModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float min_dist,
        int n_epochs,
        UwotMetric metric,
        float* embedding,
        uwot_progress_callback progress_callback) {

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
            model->metric = metric;

            // Store and normalize input data
            std::vector<float> input_data(data, data + (static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim)));

            // Compute normalization parameters
            compute_normalization(input_data, n_obs, n_dim, model->feature_means, model->feature_stds);
            model->use_normalization = true;

            // Normalize the data
            std::vector<float> normalized_data;
            normalize_data(input_data, normalized_data, n_obs, n_dim, model->feature_means, model->feature_stds);

            // Create HNSW index for normalized data
            model->l2_space = std::make_unique<hnswlib::L2Space>(n_dim);
            model->ann_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                model->l2_space.get(), n_obs, 16, 200);

            // Add all points to HNSW index
            for (int i = 0; i < n_obs; i++) {
                model->ann_index->addPoint(
                    &normalized_data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],
                    static_cast<hnswlib::labeltype>(i));
            }

            printf("[HNSW] Built index with %d points in %dD space\n", n_obs, n_dim);

            // Compute comprehensive neighbor statistics
            compute_neighbor_statistics(model, normalized_data);

            // Build k-NN graph using specified metric on original (unnormalized) data
            std::vector<int> nn_indices;
            std::vector<double> nn_distances;
            build_knn_graph(input_data, n_obs, n_dim, n_neighbors, metric,
                nn_indices, nn_distances);

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

            // Convert to edge format for optimization
            convert_to_edges(nn_indices, nn_weights, n_obs, n_neighbors,
                model->positive_head, model->positive_tail, model->positive_weights);

            // Store k-NN data for transform (flattened format)
            model->nn_indices = nn_indices;
            model->nn_distances.resize(nn_distances.size());
            model->nn_weights.resize(nn_weights.size());
            for (size_t i = 0; i < nn_distances.size(); i++) {
                model->nn_distances[i] = static_cast<float>(nn_distances[i]);
                model->nn_weights[i] = static_cast<float>(nn_weights[i]);
            }

            // Initialize embedding
            model->embedding.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim));
            std::mt19937 gen(42);
            std::normal_distribution<float> dist(0.0f, 1e-4f);
            for (size_t i = 0; i < static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim); i++) {
                model->embedding[i] = dist(gen);
            }

            // Set up UMAP parameters
            model->a = 1.929f;
            model->b = 0.7915f;
            if (min_dist != 0.1f) {
                // Adjust parameters based on min_dist (simplified)
                model->a = 1.0f / (1.0f + model->a * std::pow(min_dist, 2.0f * model->b));
                model->b = 1.0f;
            }

            // Direct UMAP optimization implementation with progress reporting
            const float learning_rate = 1.0f;
            std::mt19937 rng(42);
            std::uniform_int_distribution<size_t> vertex_dist(0, static_cast<size_t>(n_obs) - 1);

            // Progress reporting setup
            int progress_interval = std::max(1, n_epochs / 20);  // Report every 5% progress
            auto last_report_time = std::chrono::steady_clock::now();

            // Only show console output if no callback provided
            if (!progress_callback) {
                std::printf("UMAP Training Progress:\n");
                std::printf("[                    ] 0%% (Epoch 0/%d)\n", n_epochs);
                std::fflush(stdout);
            }

            for (int epoch = 0; epoch < n_epochs; epoch++) {
                float alpha = learning_rate * (1.0f - static_cast<float>(epoch) / static_cast<float>(n_epochs));

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
                        // UMAP attractive gradient: -2ab * d^(2b-2) / (1 + a*d^(2b))
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
                        }
                    }
                }

                // Progress reporting
                if (epoch % progress_interval == 0 || epoch == n_epochs - 1) {
                    float percent = (static_cast<float>(epoch + 1) / static_cast<float>(n_epochs)) * 100.0f;

                    if (progress_callback) {
                        // Use callback for C# integration
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
                        std::printf("] %d%% (Epoch %d/%d) [%lldms]",
                            percent_int, epoch + 1, n_epochs, static_cast<long long>(elapsed.count()));
                        std::fflush(stdout);

                        last_report_time = current_time;
                    }
                }
            }

            if (!progress_callback) {
                std::printf("\nTraining completed!\n");
                std::fflush(stdout);
            }

            // Copy result to output
            std::memcpy(embedding, model->embedding.data(),
                static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim) * sizeof(float));

            model->is_fitted = true;
            return UWOT_SUCCESS;

        }
        catch (...) {
            return UWOT_ERROR_MEMORY;
        }
    }

    UWOT_API int uwot_transform(UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding) {

        if (!model || !model->is_fitted || !new_data || !embedding ||
            n_new_obs <= 0 || n_dim != model->n_dim) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            std::vector<float> new_embedding(static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim));

            for (int i = 0; i < n_new_obs; i++) {
                // Normalize the new data point
                std::vector<float> normalized_point(n_dim);
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    normalized_point[j] = (new_data[idx] - model->feature_means[j]) / model->feature_stds[j];
                }

                // Use HNSW to find nearest neighbors
                auto search_result = model->ann_index->searchKnn(normalized_point.data(), model->n_neighbors);

                std::vector<int> nn_indices;
                std::vector<float> nn_weights;
                float total_weight = 0.0f;

                // Extract neighbors and compute weights
                while (!search_result.empty()) {
                    auto pair = search_result.top();
                    search_result.pop();

                    int neighbor_idx = static_cast<int>(pair.second);
                    float distance = std::sqrt(pair.first); // HNSW returns squared distance
                    float weight = std::exp(-distance * distance / (2.0f * 0.1f * 0.1f)); // Gaussian weight

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

                // Initialize new point as weighted average of neighbor embeddings
                for (int d = 0; d < model->embedding_dim; d++) {
                    float coord = 0.0f;
                    for (size_t k = 0; k < nn_indices.size(); k++) {
                        coord += model->embedding[static_cast<size_t>(nn_indices[k]) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] * nn_weights[k];
                    }
                    new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] = coord;
                }
            }

            // Copy result to output
            std::memcpy(embedding, new_embedding.data(),
                static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim) * sizeof(float));

            return UWOT_SUCCESS;

        }
        catch (...) {
            return UWOT_ERROR_MEMORY;
        }
    }

    UWOT_API int uwot_transform_detailed(UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding,
        int* nn_indices,
        float* nn_distances,
        float* confidence_score,
        int* outlier_level,
        float* percentile_rank,
        float* z_score) {

        if (!model || !model->is_fitted || !new_data || !embedding ||
            n_new_obs <= 0 || n_dim != model->n_dim) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            std::vector<float> new_embedding(static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim));

            for (int i = 0; i < n_new_obs; i++) {
                // Normalize the new data point
                std::vector<float> normalized_point(n_dim);
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    normalized_point[j] = (new_data[idx] - model->feature_means[j]) / model->feature_stds[j];
                }

                // Use HNSW to find nearest neighbors
                auto search_result = model->ann_index->searchKnn(normalized_point.data(), model->n_neighbors);

                std::vector<int> neighbors;
                std::vector<float> distances;
                std::vector<float> weights;
                float total_weight = 0.0f;

                // Extract neighbors and compute detailed statistics
                while (!search_result.empty()) {
                    auto pair = search_result.top();
                    search_result.pop();

                    int neighbor_idx = static_cast<int>(pair.second);
                    float distance = std::sqrt(pair.first); // HNSW returns squared distance
                    float weight = std::exp(-distance * distance / (2.0f * 0.1f * 0.1f));

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

                    // Confidence score (inverse of normalized distance)
                    if (confidence_score) {
                        float normalized_dist = (min_distance - model->min_neighbor_distance) /
                            (model->p95_neighbor_distance - model->min_neighbor_distance + 1e-8f);
                        confidence_score[i] = std::max(0.0f, std::min(1.0f, 1.0f - normalized_dist));
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

                    // Percentile rank (0-100)
                    if (percentile_rank) {
                        if (min_distance <= model->min_neighbor_distance) {
                            percentile_rank[i] = 0.0f;
                        }
                        else if (min_distance >= model->p99_neighbor_distance) {
                            percentile_rank[i] = 99.0f;
                        }
                        else {
                            // Linear interpolation between key percentiles
                            float p95_range = model->p95_neighbor_distance - model->min_neighbor_distance;
                            if (min_distance <= model->p95_neighbor_distance) {
                                percentile_rank[i] = 95.0f * (min_distance - model->min_neighbor_distance) / (p95_range + 1e-8f);
                            }
                            else {
                                float p99_range = model->p99_neighbor_distance - model->p95_neighbor_distance;
                                percentile_rank[i] = 95.0f + 4.0f * (min_distance - model->p95_neighbor_distance) / (p99_range + 1e-8f);
                            }
                        }
                    }

                    // Z-score relative to training data
                    if (z_score) {
                        z_score[i] = (min_distance - model->mean_neighbor_distance) / (model->std_neighbor_distance + 1e-8f);
                    }
                }

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

            // Copy embedding result to output
            std::memcpy(embedding, new_embedding.data(),
                static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim) * sizeof(float));

            return UWOT_SUCCESS;

        }
        catch (...) {
            return UWOT_ERROR_MEMORY;
        }
    }

    UWOT_API int uwot_save_model(UwotModel* model, const char* filename) {
        if (!model || !model->is_fitted || !filename) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                return UWOT_ERROR_FILE_IO;
            }

            // Write header
            const char* magic = "UMAP";
            file.write(magic, 4);
            int version = 3; // Increment version for HNSW format
            file.write(reinterpret_cast<const char*>(&version), sizeof(int));

            // Write model parameters
            file.write(reinterpret_cast<const char*>(&model->n_vertices), sizeof(int));
            file.write(reinterpret_cast<const char*>(&model->n_dim), sizeof(int));
            file.write(reinterpret_cast<const char*>(&model->embedding_dim), sizeof(int));
            file.write(reinterpret_cast<const char*>(&model->n_neighbors), sizeof(int));
            file.write(reinterpret_cast<const char*>(&model->min_dist), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->metric), sizeof(UwotMetric));
            file.write(reinterpret_cast<const char*>(&model->a), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->b), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->use_normalization), sizeof(bool));

            // Write normalization parameters
            size_t means_size = model->feature_means.size();
            file.write(reinterpret_cast<const char*>(&means_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(model->feature_means.data()),
                means_size * sizeof(float));

            size_t stds_size = model->feature_stds.size();
            file.write(reinterpret_cast<const char*>(&stds_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(model->feature_stds.data()),
                stds_size * sizeof(float));

            // Write neighbor statistics
            file.write(reinterpret_cast<const char*>(&model->min_neighbor_distance), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->mean_neighbor_distance), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->std_neighbor_distance), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->p95_neighbor_distance), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->p99_neighbor_distance), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->mild_outlier_threshold), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->extreme_outlier_threshold), sizeof(float));

            // Write embedding
            size_t embedding_size = model->embedding.size();
            file.write(reinterpret_cast<const char*>(&embedding_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(model->embedding.data()),
                embedding_size * sizeof(float));

            // Save HNSW index directly to stream (no temporary files)
            if (model->ann_index) {
                try {
                    // Capture current position to calculate size later
                    std::streampos hnsw_size_pos = file.tellp();
                    size_t placeholder_size = 0;
                    file.write(reinterpret_cast<const char*>(&placeholder_size), sizeof(size_t));

                    std::streampos hnsw_data_start = file.tellp();

                    // Save HNSW index data directly to our stream using the same logic as saveIndex
                    save_hnsw_to_stream(file, model->ann_index.get());

                    std::streampos hnsw_data_end = file.tellp();

                    // Calculate actual size and update the placeholder
                    size_t actual_hnsw_size = static_cast<size_t>(hnsw_data_end - hnsw_data_start);
                    file.seekp(hnsw_size_pos);
                    file.write(reinterpret_cast<const char*>(&actual_hnsw_size), sizeof(size_t));
                    file.seekp(hnsw_data_end);

                } catch (...) {
                    // Error saving HNSW, write zero size
                    size_t hnsw_size = 0;
                    file.write(reinterpret_cast<const char*>(&hnsw_size), sizeof(size_t));
                }
            }
            else {
                size_t hnsw_size = 0;
                file.write(reinterpret_cast<const char*>(&hnsw_size), sizeof(size_t));
            }

            file.close();
            return UWOT_SUCCESS;

        }
        catch (...) {
            return UWOT_ERROR_FILE_IO;
        }
    }

    UWOT_API UwotModel* uwot_load_model(const char* filename) {
        if (!filename) {
            return nullptr;
        }

        try {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                return nullptr;
            }

            // Read and verify header
            char magic[5] = { 0 };
            file.read(magic, 4);
            if (std::string(magic) != "UMAP") {
                file.close();
                return nullptr;
            }

            int version;
            file.read(reinterpret_cast<char*>(&version), sizeof(int));
            if (version != 1 && version != 2 && version != 3) { // Support multiple versions
                file.close();
                return nullptr;
            }

            UwotModel* model = new UwotModel();

            // Read model parameters
            file.read(reinterpret_cast<char*>(&model->n_vertices), sizeof(int));
            file.read(reinterpret_cast<char*>(&model->n_dim), sizeof(int));
            file.read(reinterpret_cast<char*>(&model->embedding_dim), sizeof(int));
            file.read(reinterpret_cast<char*>(&model->n_neighbors), sizeof(int));
            file.read(reinterpret_cast<char*>(&model->min_dist), sizeof(float));

            if (version >= 2) {
                file.read(reinterpret_cast<char*>(&model->metric), sizeof(UwotMetric));
            }
            else {
                model->metric = UWOT_METRIC_EUCLIDEAN;
            }

            file.read(reinterpret_cast<char*>(&model->a), sizeof(float));
            file.read(reinterpret_cast<char*>(&model->b), sizeof(float));

            if (version >= 3) {
                file.read(reinterpret_cast<char*>(&model->use_normalization), sizeof(bool));

                // Read normalization parameters
                size_t means_size;
                file.read(reinterpret_cast<char*>(&means_size), sizeof(size_t));
                model->feature_means.resize(means_size);
                file.read(reinterpret_cast<char*>(model->feature_means.data()),
                    means_size * sizeof(float));

                size_t stds_size;
                file.read(reinterpret_cast<char*>(&stds_size), sizeof(size_t));
                model->feature_stds.resize(stds_size);
                file.read(reinterpret_cast<char*>(model->feature_stds.data()),
                    stds_size * sizeof(float));

                // Read neighbor statistics
                file.read(reinterpret_cast<char*>(&model->min_neighbor_distance), sizeof(float));
                file.read(reinterpret_cast<char*>(&model->mean_neighbor_distance), sizeof(float));
                file.read(reinterpret_cast<char*>(&model->std_neighbor_distance), sizeof(float));
                file.read(reinterpret_cast<char*>(&model->p95_neighbor_distance), sizeof(float));
                file.read(reinterpret_cast<char*>(&model->p99_neighbor_distance), sizeof(float));
                file.read(reinterpret_cast<char*>(&model->mild_outlier_threshold), sizeof(float));
                file.read(reinterpret_cast<char*>(&model->extreme_outlier_threshold), sizeof(float));
            }
            else {
                // For older versions, set defaults
                model->use_normalization = false;
                model->min_neighbor_distance = 0.0f;
                model->mean_neighbor_distance = 0.0f;
                model->std_neighbor_distance = 0.0f;
                model->p95_neighbor_distance = 0.0f;
                model->p99_neighbor_distance = 0.0f;
                model->mild_outlier_threshold = 0.0f;
                model->extreme_outlier_threshold = 0.0f;
            }

            // Read embedding
            size_t embedding_size;
            file.read(reinterpret_cast<char*>(&embedding_size), sizeof(size_t));
            model->embedding.resize(embedding_size);
            file.read(reinterpret_cast<char*>(model->embedding.data()),
                embedding_size * sizeof(float));

            if (version >= 3) {
                // Read HNSW index
                size_t hnsw_size;
                file.read(reinterpret_cast<char*>(&hnsw_size), sizeof(size_t));

                if (hnsw_size > 0) {
                    try {
                        // Load HNSW index directly from stream (no temporary files)
                        model->l2_space = std::make_unique<hnswlib::L2Space>(model->n_dim);
                        model->ann_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                            model->l2_space.get());

                        // Load HNSW data directly from our stream using the same logic as loadIndex
                        load_hnsw_from_stream(file, model->ann_index.get(), model->l2_space.get(), hnsw_size);

                    } catch (...) {
                        // Error loading HNSW, clean up and continue without index
                        model->l2_space = nullptr;
                        model->ann_index = nullptr;

                        // Skip remaining HNSW data in file
                        try {
                            file.seekg(static_cast<std::streamoff>(hnsw_size), std::ios::cur);
                        } catch (...) {
                            // If seek fails, we're in trouble
                        }
                    }
                }
            }
            else {
                // For older versions without HNSW, create empty structures
                model->l2_space = nullptr;
                model->ann_index = nullptr;
            }

            model->is_fitted = true;
            file.close();
            return model;

        }
        catch (...) {
            return nullptr;
        }
    }

    UWOT_API int uwot_get_model_info(UwotModel* model,
        int* n_vertices,
        int* n_dim,
        int* embedding_dim,
        int* n_neighbors,
        float* min_dist,
        UwotMetric* metric) {
        if (!model) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        if (n_vertices) *n_vertices = model->n_vertices;
        if (n_dim) *n_dim = model->n_dim;
        if (embedding_dim) *embedding_dim = model->embedding_dim;
        if (n_neighbors) *n_neighbors = model->n_neighbors;
        if (min_dist) *min_dist = model->min_dist;
        if (metric) *metric = model->metric;

        return UWOT_SUCCESS;
    }

    UWOT_API void uwot_destroy(UwotModel* model) {
        delete model;
    }

    UWOT_API const char* uwot_get_error_message(int error_code) {
        switch (error_code) {
        case UWOT_SUCCESS:
            return "Success";
        case UWOT_ERROR_INVALID_PARAMS:
            return "Invalid parameters";
        case UWOT_ERROR_MEMORY:
            return "Memory allocation error";
        case UWOT_ERROR_NOT_IMPLEMENTED:
            return "Feature not implemented";
        case UWOT_ERROR_FILE_IO:
            return "File I/O error";
        case UWOT_ERROR_MODEL_NOT_FITTED:
            return "Model not fitted";
        case UWOT_ERROR_INVALID_MODEL_FILE:
            return "Invalid model file";
        default:
            return "Unknown error";
        }
    }

    UWOT_API const char* uwot_get_metric_name(UwotMetric metric) {
        switch (metric) {
        case UWOT_METRIC_EUCLIDEAN:
            return "euclidean";
        case UWOT_METRIC_COSINE:
            return "cosine";
        case UWOT_METRIC_MANHATTAN:
            return "manhattan";
        case UWOT_METRIC_CORRELATION:
            return "correlation";
        case UWOT_METRIC_HAMMING:
            return "hamming";
        default:
            return "unknown";
        }
    }

    UWOT_API int uwot_get_embedding_dim(UwotModel* model) {
        return model ? model->embedding_dim : -1;
    }

    UWOT_API int uwot_get_n_vertices(UwotModel* model) {
        return model ? model->n_vertices : -1;
    }

    UWOT_API int uwot_is_fitted(UwotModel* model) {
        return model ? (model->is_fitted ? 1 : 0) : 0;
    }

} // extern "C"