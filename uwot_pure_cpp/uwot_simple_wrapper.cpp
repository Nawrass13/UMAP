#include "uwot_simple_wrapper.h"
#include <cmath>  // Add this line for sqrt, etc.
#include "optimize.h"
#include "gradient.h" 
#include "coords.h"
#include "sampler.h"
#include "perplexity.h"
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <fstream>
#include <sstream>

struct UwotModel {
    // Model parameters
    int n_vertices;
    int n_dim;
    int embedding_dim;
    int n_neighbors;
    float min_dist;
    float a, b; // UMAP curve parameters
    bool is_fitted;

    // Training data (needed for transform)
    std::vector<float> training_data;

    // Graph structure
    std::vector<unsigned int> positive_head;
    std::vector<unsigned int> positive_tail;
    std::vector<float> positive_weights;
    std::vector<unsigned int> epochs_per_sample;

    // Final embedding
    std::vector<float> embedding;

    // k-NN structure for transformation
    std::vector<std::vector<int>> knn_indices;
    std::vector<std::vector<float>> knn_distances;

    UwotModel() : n_vertices(0), n_dim(0), embedding_dim(2), n_neighbors(15),
        min_dist(0.1f), a(1.929f), b(0.7915f), is_fitted(false) {
    }
};

// Standalone smooth k-NN implementation without Rcpp dependencies
namespace standalone_smooth_knn {

    // Compute smooth k-nearest neighbor weights using perplexity-based approach
    void compute_smooth_knn_weights(const std::vector<std::vector<float>>& distances,
        std::vector<std::vector<float>>& weights,
        float target_perplexity = 15.0f,
        int max_iter = 200,
        float tolerance = 1e-5f) {
        int n = static_cast<int>(distances.size());
        weights.resize(n);

        for (int i = 0; i < n; i++) {
            int k = static_cast<int>(distances[i].size());
            weights[i].resize(k);

            if (k == 0) continue;

            // Binary search for sigma that gives target perplexity
            float sigma_min = 1e-20f;
            float sigma_max = 1e10f;
            float sigma = 1.0f;

            for (int iter = 0; iter < max_iter; iter++) {
                // Compute P_j|i with current sigma
                std::vector<float> p_conditional(k);
                float sum_p = 0.0f;

                for (int j = 0; j < k; j++) {
                    p_conditional[j] = std::exp(-distances[i][j] * distances[i][j] / (2.0f * sigma * sigma));
                    sum_p += p_conditional[j];
                }

                // Normalize
                if (sum_p > 1e-20f) {
                    for (int j = 0; j < k; j++) {
                        p_conditional[j] /= sum_p;
                    }
                }
                else {
                    // Handle degenerate case
                    for (int j = 0; j < k; j++) {
                        p_conditional[j] = 1.0f / k;
                    }
                }

                // Compute perplexity
                float entropy = 0.0f;
                for (int j = 0; j < k; j++) {
                    if (p_conditional[j] > 1e-20f) {
                        entropy -= p_conditional[j] * std::log2(p_conditional[j]);
                    }
                }
                float perplexity = std::pow(2.0f, entropy);

                // Check convergence
                if (std::abs(perplexity - target_perplexity) < tolerance) {
                    weights[i] = p_conditional;
                    break;
                }

                // Adjust sigma
                if (perplexity > target_perplexity) {
                    sigma_max = sigma;
                    sigma = (sigma + sigma_min) / 2.0f;
                }
                else {
                    sigma_min = sigma;
                    sigma = (sigma + sigma_max) / 2.0f;
                }

                // Final iteration fallback
                if (iter == max_iter - 1) {
                    weights[i] = p_conditional;
                }
            }
        }
    }

    // Symmetrize the probability matrix
    void symmetrize_matrix(const std::vector<std::vector<int>>& indices,
        const std::vector<std::vector<float>>& weights,
        std::vector<unsigned int>& head,
        std::vector<unsigned int>& tail,
        std::vector<float>& symmetric_weights) {

        int n = static_cast<int>(indices.size());

        // Build sparse matrix representation
        std::map<std::pair<int, int>, float> edge_map;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < static_cast<int>(indices[i].size()); j++) {
                int neighbor = indices[i][j];
                float weight = weights[i][j];

                // Add both directions
                edge_map[{i, neighbor}] += weight;
                edge_map[{neighbor, i}] += weight;
            }
        }

        // Convert to edge list and symmetrize
        head.clear();
        tail.clear();
        symmetric_weights.clear();

        for (const auto& edge : edge_map) {
            int i = edge.first.first;
            int j = edge.first.second;

            if (i < j) { // Avoid duplicates
                float sym_weight = edge.second / 2.0f; // Average the weights

                head.push_back(static_cast<unsigned int>(i));
                tail.push_back(static_cast<unsigned int>(j));
                symmetric_weights.push_back(sym_weight);
            }
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
        int n_neighbors,
        float min_dist,
        int n_epochs,
        float* embedding) {

        if (!model || !data || !embedding || n_obs <= 0 || n_dim <= 0 ||
            n_neighbors <= 0 || n_epochs <= 0) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            model->n_vertices = n_obs;
            model->n_dim = n_dim;
            model->embedding_dim = 2; // Default to 2D embedding
            model->n_neighbors = n_neighbors;
            model->min_dist = min_dist;

            // Store training data for future transformations
            model->training_data.assign(data, data + (n_obs * n_dim));

            // Adjust UMAP curve parameters based on min_dist
            if (min_dist != 0.1f) {
                model->a = 1.0f / (1.0f + model->a * std::pow(min_dist, 2.0f * model->b));
                model->b = 1.0f;
            }

            // Step 1: Build k-nearest neighbor graph
            model->knn_indices.resize(n_obs);
            model->knn_distances.resize(n_obs);

            // Simple brute force k-NN
            for (int i = 0; i < n_obs; i++) {
                std::vector<std::pair<float, int>> distances;

                for (int j = 0; j < n_obs; j++) {
                    if (i == j) continue;

                    float dist = 0.0f;
                    for (int k = 0; k < n_dim; k++) {
                        float diff = data[i * n_dim + k] - data[j * n_dim + k];
                        dist += diff * diff;
                    }
                    dist = std::sqrt(dist);
                    distances.push_back({ dist, j });
                }

                // Get k nearest neighbors
                std::partial_sort(distances.begin(),
                    distances.begin() + n_neighbors,
                    distances.end());

                model->knn_indices[i].resize(n_neighbors);
                model->knn_distances[i].resize(n_neighbors);
                for (int k = 0; k < n_neighbors; k++) {
                    model->knn_indices[i][k] = distances[k].second;
                    model->knn_distances[i][k] = distances[k].first;
                }
            }

            // Step 2: Compute smooth k-NN weights using standalone implementation
            std::vector<std::vector<float>> smooth_weights;
            standalone_smooth_knn::compute_smooth_knn_weights(model->knn_distances, smooth_weights, 15.0f);

            // Step 3: Symmetrize to get final graph
            standalone_smooth_knn::symmetrize_matrix(model->knn_indices, smooth_weights,
                model->positive_head,
                model->positive_tail,
                model->positive_weights);

            // Step 4: Initialize embedding
            model->embedding.resize(n_obs * model->embedding_dim);
            std::mt19937 gen(42); // Fixed seed for reproducibility
            std::normal_distribution<float> dist(0.0f, 1e-4f);

            for (int i = 0; i < n_obs * model->embedding_dim; i++) {
                model->embedding[i] = dist(gen);
            }

            // Step 5: Compute epochs per sample
            model->epochs_per_sample.resize(model->positive_weights.size());
            if (!model->positive_weights.empty()) {
                float max_weight = *std::max_element(model->positive_weights.begin(),
                    model->positive_weights.end());

                for (size_t i = 0; i < model->positive_weights.size(); i++) {
                    model->epochs_per_sample[i] = static_cast<unsigned int>(
                        std::max(1.0f, n_epochs * model->positive_weights[i] / max_weight));
                }
            }

            // Step 6: Optimization (simplified gradient descent)
            float learning_rate = 1.0f;
            float a = model->a;
            float b = model->b;

            for (int epoch = 0; epoch < n_epochs; epoch++) {
                float alpha = learning_rate * (1.0f - static_cast<float>(epoch) / n_epochs);

                // Process positive samples
                for (size_t edge = 0; edge < model->positive_head.size(); edge++) {
                    if (model->epochs_per_sample[edge] > 0 &&
                        epoch % model->epochs_per_sample[edge] == 0) {

                        unsigned int i = model->positive_head[edge];
                        unsigned int j = model->positive_tail[edge];

                        // Compute current distance in embedding space
                        float dist_sq = 0.0f;
                        for (int d = 0; d < model->embedding_dim; d++) {
                            float diff = model->embedding[i * model->embedding_dim + d] -
                                model->embedding[j * model->embedding_dim + d];
                            dist_sq += diff * diff;
                        }

                        // Avoid division by zero
                        dist_sq = std::max(dist_sq, 1e-12f);

                        // Attractive force gradient
                        float grad_coeff = 2.0f * a * b * std::pow(dist_sq, b - 1.0f) /
                            (a * std::pow(dist_sq, b) + 1.0f);

                        // Apply gradient
                        for (int d = 0; d < model->embedding_dim; d++) {
                            float diff = model->embedding[i * model->embedding_dim + d] -
                                model->embedding[j * model->embedding_dim + d];
                            float grad = grad_coeff * diff;

                            model->embedding[i * model->embedding_dim + d] -= alpha * grad;
                            model->embedding[j * model->embedding_dim + d] += alpha * grad;
                        }
                    }
                }

                // Negative sampling
                if (epoch % 5 == 0) {
                    std::uniform_int_distribution<int> point_dist(0, n_obs - 1);

                    for (int neg_sample = 0; neg_sample < n_obs / 10; neg_sample++) {
                        int i = point_dist(gen);
                        int j = point_dist(gen);
                        if (i == j) continue;

                        float dist_sq = 0.0f;
                        for (int d = 0; d < model->embedding_dim; d++) {
                            float diff = model->embedding[i * model->embedding_dim + d] -
                                model->embedding[j * model->embedding_dim + d];
                            dist_sq += diff * diff;
                        }

                        if (dist_sq > 0.0f) {
                            // Repulsive force gradient
                            float grad_coeff = 2.0f * b / (0.001f + dist_sq) /
                                (a * std::pow(dist_sq, b) + 1.0f);

                            for (int d = 0; d < model->embedding_dim; d++) {
                                float diff = model->embedding[i * model->embedding_dim + d] -
                                    model->embedding[j * model->embedding_dim + d];
                                float grad = grad_coeff * diff;

                                model->embedding[i * model->embedding_dim + d] += alpha * grad;
                                model->embedding[j * model->embedding_dim + d] -= alpha * grad;
                            }
                        }
                    }
                }
            }

            // Copy result to output
            std::memcpy(embedding, model->embedding.data(),
                n_obs * model->embedding_dim * sizeof(float));

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
            // For each new data point, find its position in the existing embedding
            for (int i = 0; i < n_new_obs; i++) {
                // Find k nearest neighbors in training data
                std::vector<std::pair<float, int>> distances;

                for (int j = 0; j < model->n_vertices; j++) {
                    float dist = 0.0f;
                    for (int k = 0; k < n_dim; k++) {
                        float diff = new_data[i * n_dim + k] - model->training_data[j * n_dim + k];
                        dist += diff * diff;
                    }
                    dist = std::sqrt(dist);
                    distances.push_back({ dist, j });
                }

                // Get k nearest neighbors
                int k = std::min(model->n_neighbors, static_cast<int>(distances.size()));
                std::partial_sort(distances.begin(), distances.begin() + k, distances.end());

                // Weighted average of neighbor embeddings
                float total_weight = 0.0f;
                float embed_x = 0.0f, embed_y = 0.0f;

                for (int j = 0; j < k; j++) {
                    int neighbor_idx = distances[j].second;
                    float dist = distances[j].first;

                    // Use inverse distance weighting with exponential decay
                    float weight = std::exp(-dist * dist / (2.0f * 0.1f * 0.1f));

                    embed_x += weight * model->embedding[neighbor_idx * model->embedding_dim];
                    embed_y += weight * model->embedding[neighbor_idx * model->embedding_dim + 1];
                    total_weight += weight;
                }

                if (total_weight > 0.0f) {
                    embedding[i * model->embedding_dim] = embed_x / total_weight;
                    embedding[i * model->embedding_dim + 1] = embed_y / total_weight;
                }
                else {
                    // Fallback: use nearest neighbor
                    int nearest = distances[0].second;
                    embedding[i * model->embedding_dim] = model->embedding[nearest * model->embedding_dim];
                    embedding[i * model->embedding_dim + 1] = model->embedding[nearest * model->embedding_dim + 1];
                }
            }

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

            int version = 1;
            file.write(reinterpret_cast<const char*>(&version), sizeof(int));

            // Write model parameters
            file.write(reinterpret_cast<const char*>(&model->n_vertices), sizeof(int));
            file.write(reinterpret_cast<const char*>(&model->n_dim), sizeof(int));
            file.write(reinterpret_cast<const char*>(&model->embedding_dim), sizeof(int));
            file.write(reinterpret_cast<const char*>(&model->n_neighbors), sizeof(int));
            file.write(reinterpret_cast<const char*>(&model->min_dist), sizeof(float));
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

            // Write k-NN indices
            file.write(reinterpret_cast<const char*>(&model->n_vertices), sizeof(int));
            for (int i = 0; i < model->n_vertices; i++) {
                size_t knn_size = model->knn_indices[i].size();
                file.write(reinterpret_cast<const char*>(&knn_size), sizeof(size_t));
                file.write(reinterpret_cast<const char*>(model->knn_indices[i].data()),
                    knn_size * sizeof(int));
            }

            // Write k-NN distances
            for (int i = 0; i < model->n_vertices; i++) {
                size_t knn_size = model->knn_distances[i].size();
                file.write(reinterpret_cast<const char*>(&knn_size), sizeof(size_t));
                file.write(reinterpret_cast<const char*>(model->knn_distances[i].data()),
                    knn_size * sizeof(float));
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
            if (version != 1) {
                file.close();
                return nullptr;
            }

            // Create new model
            UwotModel* model = new UwotModel();

            // Read model parameters
            file.read(reinterpret_cast<char*>(&model->n_vertices), sizeof(int));
            file.read(reinterpret_cast<char*>(&model->n_dim), sizeof(int));
            file.read(reinterpret_cast<char*>(&model->embedding_dim), sizeof(int));
            file.read(reinterpret_cast<char*>(&model->n_neighbors), sizeof(int));
            file.read(reinterpret_cast<char*>(&model->min_dist), sizeof(float));
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

            // Read k-NN indices
            int n_vertices_check;
            file.read(reinterpret_cast<char*>(&n_vertices_check), sizeof(int));
            if (n_vertices_check != model->n_vertices) {
                delete model;
                file.close();
                return nullptr;
            }

            model->knn_indices.resize(model->n_vertices);
            for (int i = 0; i < model->n_vertices; i++) {
                size_t knn_size;
                file.read(reinterpret_cast<char*>(&knn_size), sizeof(size_t));
                model->knn_indices[i].resize(knn_size);
                file.read(reinterpret_cast<char*>(model->knn_indices[i].data()),
                    knn_size * sizeof(int));
            }

            // Read k-NN distances
            model->knn_distances.resize(model->n_vertices);
            for (int i = 0; i < model->n_vertices; i++) {
                size_t knn_size;
                file.read(reinterpret_cast<char*>(&knn_size), sizeof(size_t));
                model->knn_distances[i].resize(knn_size);
                file.read(reinterpret_cast<char*>(model->knn_distances[i].data()),
                    knn_size * sizeof(float));
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
        float* min_dist) {
        if (!model) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        if (n_vertices) *n_vertices = model->n_vertices;
        if (n_dim) *n_dim = model->n_dim;
        if (embedding_dim) *embedding_dim = model->embedding_dim;
        if (n_neighbors) *n_neighbors = model->n_neighbors;
        if (min_dist) *min_dist = model->min_dist;

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