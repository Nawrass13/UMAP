#include "uwot_simple_wrapper.h"
#include "smooth_knn.h"
#include "optimize.h"
#include "gradient.h" 
#include "transform.h"
#include "update.h"
#include "epoch.h"
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
#include <cstdio>

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

    // Training data (needed for transform)
    std::vector<float> training_data;

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

    UwotModel() : n_vertices(0), n_dim(0), embedding_dim(2), n_neighbors(15),
        min_dist(0.1f), metric(UWOT_METRIC_EUCLIDEAN), a(1.929f), b(0.7915f), is_fitted(false) {
    }
};

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

            // Store training data
            std::vector<float> input_data(data, data + (static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim)));
            model->training_data = input_data;

            // Build k-NN graph using specified metric
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
            // Build k-NN from new data to training data
            std::vector<int> new_nn_indices(static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->n_neighbors));
            std::vector<float> new_nn_weights(static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->n_neighbors));

            for (int i = 0; i < n_new_obs; i++) {
                std::vector<std::pair<float, int>> distances;

                // Find nearest neighbors in training data
                for (int j = 0; j < model->n_vertices; j++) {
                    float dist = distance_metrics::compute_distance(
                        &new_data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],
                        &model->training_data[static_cast<size_t>(j) * static_cast<size_t>(n_dim)],
                        n_dim, model->metric);
                    distances.push_back({ dist, j });
                }

                std::partial_sort(distances.begin(),
                    distances.begin() + model->n_neighbors,
                    distances.end());

                // Store indices and compute simple inverse distance weights
                float total_weight = 0.0f;
                for (int k = 0; k < model->n_neighbors; k++) {
                    new_nn_indices[static_cast<size_t>(i) * static_cast<size_t>(model->n_neighbors) + static_cast<size_t>(k)] = distances[static_cast<size_t>(k)].second;
                    float weight = std::exp(-distances[static_cast<size_t>(k)].first * distances[static_cast<size_t>(k)].first / (2.0f * 0.1f * 0.1f));
                    new_nn_weights[static_cast<size_t>(i) * static_cast<size_t>(model->n_neighbors) + static_cast<size_t>(k)] = weight;
                    total_weight += weight;
                }

                // Normalize weights
                if (total_weight > 0.0f) {
                    for (int k = 0; k < model->n_neighbors; k++) {
                        new_nn_weights[static_cast<size_t>(i) * static_cast<size_t>(model->n_neighbors) + static_cast<size_t>(k)] /= total_weight;
                    }
                }
            }

            // Use uwot transform to initialize new points
            std::vector<float> new_embedding(static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim));
            uwot::init_by_mean(0, static_cast<std::size_t>(n_new_obs), static_cast<std::size_t>(model->embedding_dim),
                static_cast<std::size_t>(model->n_neighbors),
                new_nn_indices, new_nn_weights,
                static_cast<std::size_t>(n_new_obs),
                model->embedding,
                static_cast<std::size_t>(model->n_vertices),
                new_embedding);

            // Copy result to output
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
            int version = 2;
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

            // Write training data
            size_t training_size = model->training_data.size();
            file.write(reinterpret_cast<const char*>(&training_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(model->training_data.data()),
                training_size * sizeof(float));

            // Write embedding
            size_t embedding_size = model->embedding.size();
            file.write(reinterpret_cast<const char*>(&embedding_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(model->embedding.data()),
                embedding_size * sizeof(float));

            // Write k-NN data for transform
            size_t nn_indices_size = model->nn_indices.size();
            file.write(reinterpret_cast<const char*>(&nn_indices_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(model->nn_indices.data()),
                nn_indices_size * sizeof(int));

            size_t nn_distances_size = model->nn_distances.size();
            file.write(reinterpret_cast<const char*>(&nn_distances_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(model->nn_distances.data()),
                nn_distances_size * sizeof(float));

            size_t nn_weights_size = model->nn_weights.size();
            file.write(reinterpret_cast<const char*>(&nn_weights_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(model->nn_weights.data()),
                nn_weights_size * sizeof(float));

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
            if (version != 1 && version != 2) {
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

            // Read training data
            size_t training_size;
            file.read(reinterpret_cast<char*>(&training_size), sizeof(size_t));
            model->training_data.resize(training_size);
            file.read(reinterpret_cast<char*>(model->training_data.data()),
                training_size * sizeof(float));

            // Read embedding
            size_t embedding_size;
            file.read(reinterpret_cast<char*>(&embedding_size), sizeof(size_t));
            model->embedding.resize(embedding_size);
            file.read(reinterpret_cast<char*>(model->embedding.data()),
                embedding_size * sizeof(float));

            // Read k-NN data for transform
            size_t nn_indices_size;
            file.read(reinterpret_cast<char*>(&nn_indices_size), sizeof(size_t));
            model->nn_indices.resize(nn_indices_size);
            file.read(reinterpret_cast<char*>(model->nn_indices.data()),
                nn_indices_size * sizeof(int));

            size_t nn_distances_size;
            file.read(reinterpret_cast<char*>(&nn_distances_size), sizeof(size_t));
            model->nn_distances.resize(nn_distances_size);
            file.read(reinterpret_cast<char*>(model->nn_distances.data()),
                nn_distances_size * sizeof(float));

            size_t nn_weights_size;
            file.read(reinterpret_cast<char*>(&nn_weights_size), sizeof(size_t));
            model->nn_weights.resize(nn_weights_size);
            file.read(reinterpret_cast<char*>(model->nn_weights.data()),
                nn_weights_size * sizeof(float));

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