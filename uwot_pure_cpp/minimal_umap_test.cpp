#include "uwot_simple_wrapper.h"
#include <cmath>
#include "optimize.h"
#include "gradient.h"
#include "coords.h"
#include "sampler.h"
// Remove smooth_knn.h - it has Rcpp dependencies
// #include "smooth_knn.h"
#include "perplexity.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <map>

// Move template functions to global scope (outside of any local class/function)
template<typename T>
T euclidean_distance_squared(const std::vector<T>& a, const std::vector<T>& b) {
    T dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        T diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

template<typename T>
void normalize_vector(std::vector<T>& vec) {
    T sum = 0;
    for (const auto& val : vec) {
        sum += val;
    }
    if (sum > 0) {
        for (auto& val : vec) {
            val /= sum;
        }
    }
}

// Helper class for k-NN computation
class SimpleKNN {
public:
    static void compute_knn(const std::vector<std::vector<float>>& data,
        int k,
        std::vector<std::vector<int>>& indices,
        std::vector<std::vector<float>>& distances) {
        int n = static_cast<int>(data.size());
        indices.resize(n);
        distances.resize(n);

        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<float, int>> neighbors;

            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    float dist = std::sqrt(euclidean_distance_squared(data[i], data[j]));
                    neighbors.emplace_back(dist, j);
                }
            }

            std::partial_sort(neighbors.begin(),
                neighbors.begin() + k,
                neighbors.end());

            indices[i].resize(k);
            distances[i].resize(k);

            for (int j = 0; j < k; ++j) {
                indices[i][j] = neighbors[j].second;
                distances[i][j] = neighbors[j].first;
            }
        }
    }
};

// Simple test data generator
std::vector<std::vector<float>> generate_test_data(int n_samples, int n_features, int random_seed = 42) {
    std::mt19937 gen(random_seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> data(n_samples, std::vector<float>(n_features));

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            data[i][j] = dist(gen);
        }
    }

    return data;
}

int main() {
    std::cout << "Testing UMAP C++ wrapper..." << std::endl;

    // Generate test data
    const int n_samples = 100;
    const int n_features = 10;
    const int n_neighbors = 15;
    const int n_epochs = 200;
    const float min_dist = 0.1f;
    const float spread = 1.0f;

    std::cout << "Generating test data: " << n_samples << " samples, "
        << n_features << " features" << std::endl;

    auto test_data = generate_test_data(n_samples, n_features);

    // Convert to flat array for C interface
    std::vector<float> flat_data(n_samples * n_features);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            flat_data[i * n_features + j] = test_data[i][j];
        }
    }

    // Test k-NN computation
    std::cout << "Computing k-NN graph..." << std::endl;
    std::vector<std::vector<int>> knn_indices;
    std::vector<std::vector<float>> knn_distances;

    SimpleKNN::compute_knn(test_data, n_neighbors, knn_indices, knn_distances);

    std::cout << "k-NN computation completed" << std::endl;
    std::cout << "First sample neighbors: ";
    for (int i = 0; i < std::min(5, n_neighbors); ++i) {
        std::cout << knn_indices[0][i] << " ";
    }
    std::cout << std::endl;

    // Test the wrapper
    std::cout << "Testing UMAP wrapper..." << std::endl;

    UwotModel* model = uwot_create();
    if (model == nullptr) {
        std::cerr << "Failed to create UMAP model" << std::endl;
        return 1;
    }

    std::cout << "UMAP model created successfully" << std::endl;

    // Prepare embedding output
    std::vector<float> embedding(n_samples * 2); // 2D embedding

    // Call fit function
    std::cout << "Running UMAP fit with " << n_epochs << " epochs..." << std::endl;
    int result = uwot_fit(model, flat_data.data(), n_samples, n_features, 2,
        n_neighbors, min_dist, spread, n_epochs, UWOT_METRIC_EUCLIDEAN, embedding.data(), 0);

    if (result == UWOT_SUCCESS) {
        std::cout << "UMAP fitting completed successfully!" << std::endl;

        // Print first few embedding coordinates
        std::cout << "First 5 embedding coordinates:" << std::endl;
        for (int i = 0; i < std::min(5, n_samples); ++i) {
            std::cout << "Sample " << i << ": ("
                << embedding[i * 2] << ", "
                << embedding[i * 2 + 1] << ")" << std::endl;
        }

        // Compute some basic statistics
        float min_x = embedding[0], max_x = embedding[0];
        float min_y = embedding[1], max_y = embedding[1];

        for (int i = 0; i < n_samples; ++i) {
            float x = embedding[i * 2];
            float y = embedding[i * 2 + 1];

            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }

        std::cout << "Embedding bounds: X=[" << min_x << ", " << max_x
            << "], Y=[" << min_y << ", " << max_y << "]" << std::endl;

        float spread_score = std::sqrt((max_x - min_x) * (max_x - min_x) + (max_y - min_y) * (max_y - min_y));
        std::cout << "Spread score (diagonal): " << spread_score << std::endl;

    }
    else {
        std::cout << "UMAP fitting failed with error code: " << result << std::endl;
        std::cout << "Error message: " << uwot_get_error_message(result) << std::endl;
    }

    // NEW: Test spread parameter functionality
    std::cout << "\n=== Testing Spread Parameter (NEW in v3.1.1) ===" << std::endl;

    const std::vector<float> test_spreads = {0.5f, 1.0f, 2.0f, 5.0f};

    for (float test_spread : test_spreads) {
        std::cout << "\nTesting spread = " << test_spread << std::endl;

        UwotModel* spread_model = uwot_create();
        std::vector<float> spread_embedding(n_samples * 2);

        int spread_result = uwot_fit(spread_model, flat_data.data(), n_samples, n_features, 2,
            n_neighbors, 0.35f, test_spread, 100, UWOT_METRIC_EUCLIDEAN, spread_embedding.data(), 0);

        if (spread_result == UWOT_SUCCESS) {
            // Calculate spread metrics
            float spread_min_x = spread_embedding[0], spread_max_x = spread_embedding[0];
            float spread_min_y = spread_embedding[1], spread_max_y = spread_embedding[1];

            for (int i = 0; i < n_samples; ++i) {
                float x = spread_embedding[i * 2];
                float y = spread_embedding[i * 2 + 1];
                spread_min_x = std::min(spread_min_x, x);
                spread_max_x = std::max(spread_max_x, x);
                spread_min_y = std::min(spread_min_y, y);
                spread_max_y = std::max(spread_max_y, y);
            }

            float diagonal = std::sqrt((spread_max_x - spread_min_x) * (spread_max_x - spread_min_x) +
                                      (spread_max_y - spread_min_y) * (spread_max_y - spread_min_y));

            std::cout << "  Bounds: X=[" << spread_min_x << ", " << spread_max_x
                     << "], Y=[" << spread_min_y << ", " << spread_max_y << "]" << std::endl;
            std::cout << "  Diagonal length: " << diagonal << " (spread effect: "
                     << (diagonal / test_spread) << ")" << std::endl;
        } else {
            std::cout << "  Failed with error: " << spread_result << std::endl;
        }

        uwot_destroy(spread_model);
    }

    std::cout << "\n✓ Spread parameter testing completed!" << std::endl;
    std::cout << "Higher spread values should produce more dispersed embeddings." << std::endl;

    // Test utility functions
    std::cout << "Model info - Vertices: " << uwot_get_n_vertices(model)
        << ", Embedding dim: " << uwot_get_embedding_dim(model) << std::endl;

    // Clean up
    uwot_destroy(model);
    std::cout << "UMAP model destroyed" << std::endl;

    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}