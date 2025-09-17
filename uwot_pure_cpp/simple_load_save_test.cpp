#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "uwot_simple_wrapper.h"

// Generate simple test data
std::vector<float> generate_data(int n_samples, int n_features) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(n_samples * n_features);
    for (int i = 0; i < n_samples * n_features; i++) {
        data[i] = dist(rng);
    }
    return data;
}

bool test_save_load(const std::string& name, bool use_quantization, bool force_exact_knn) {
    printf("\n=== %s ===\n", name.c_str());
    printf("Quantization: %s, k-NN: %s\n",
           use_quantization ? "ON" : "OFF",
           force_exact_knn ? "EXACT" : "HNSW");

    // Small dataset
    int n_samples = 50;
    int n_features = 10;
    int embedding_dim = 2;
    int n_neighbors = 5;

    auto data = generate_data(n_samples, n_features);

    // Train original model
    UwotModel* model = uwot_create();

    printf("Training original model...\n");
    int result = uwot_fit_with_progress(
        model, data.data(), n_samples, n_features,
        embedding_dim, n_neighbors, 0.1f, 1.0f, 50, // min_dist, spread, epochs
        UWOT_METRIC_EUCLIDEAN,
        nullptr, // embedding output
        nullptr, // progress callback
        force_exact_knn ? 1 : 0, // force_exact_knn
        use_quantization ? 1 : 0 // use_quantization
    );

    if (result != UWOT_SUCCESS) {
        printf("❌ Training failed: %d\n", result);
        uwot_destroy(model);
        return false;
    }

    // Test transform with original model
    std::vector<float> test_point(n_features);
    for (int i = 0; i < n_features; i++) {
        test_point[i] = data[i]; // First training sample
    }

    std::vector<float> orig_embedding(embedding_dim);
    std::vector<int> nn_indices(n_neighbors);
    std::vector<float> nn_distances(n_neighbors);
    float confidence_score;
    int outlier_level;
    float percentile_rank, z_score;

    result = uwot_transform_detailed(model, test_point.data(), 1, n_features,
                                   orig_embedding.data(), nn_indices.data(), nn_distances.data(),
                                   &confidence_score, &outlier_level, &percentile_rank, &z_score);

    if (result != UWOT_SUCCESS) {
        printf("❌ Original transform failed: %d\n", result);
        uwot_destroy(model);
        return false;
    }

    printf("Original projection: [%.6f, %.6f] conf=%.3f\n",
           orig_embedding[0], orig_embedding[1], confidence_score);

    // Save model
    std::string filename = "test_" + name + ".umap";
    printf("Saving to %s...\n", filename.c_str());

    result = uwot_save_model(model, filename.c_str());
    if (result != UWOT_SUCCESS) {
        printf("❌ Save failed: %d\n", result);
        uwot_destroy(model);
        return false;
    }

    // Destroy original
    uwot_destroy(model);

    // Load model
    printf("Loading from %s...\n", filename.c_str());
    UwotModel* loaded_model = uwot_load_model(filename.c_str());
    if (!loaded_model) {
        printf("❌ Load failed\n");
        return false;
    }

    // Test transform with loaded model
    std::vector<float> loaded_embedding(embedding_dim);

    result = uwot_transform_detailed(loaded_model, test_point.data(), 1, n_features,
                                   loaded_embedding.data(), nn_indices.data(), nn_distances.data(),
                                   &confidence_score, &outlier_level, &percentile_rank, &z_score);

    if (result != UWOT_SUCCESS) {
        printf("❌ Loaded transform failed: %d\n", result);
        uwot_destroy(loaded_model);
        return false;
    }

    printf("Loaded projection:   [%.6f, %.6f] conf=%.3f\n",
           loaded_embedding[0], loaded_embedding[1], confidence_score);

    // Compare projections
    float diff_x = std::abs(orig_embedding[0] - loaded_embedding[0]);
    float diff_y = std::abs(orig_embedding[1] - loaded_embedding[1]);
    float max_diff = std::max(diff_x, diff_y);

    printf("Differences:         [%.6f, %.6f] max=%.6f\n", diff_x, diff_y, max_diff);

    const float tolerance = use_quantization ? 0.1f : 0.001f;
    bool success = max_diff < tolerance;

    if (success) {
        printf("✅ PASSED: Projections match (diff=%.6f < %.6f)\n", max_diff, tolerance);
    } else {
        printf("❌ FAILED: Projections differ (diff=%.6f >= %.6f)\n", max_diff, tolerance);
    }

    // Cleanup
    uwot_destroy(loaded_model);
    std::remove(filename.c_str());

    return success;
}

int main() {
    printf("🧪 SIMPLE LOAD/SAVE/COMPARE TEST\n");
    printf("=================================\n");

    std::vector<bool> results;

    // Test all 4 combinations
    results.push_back(test_save_load("no_pq_hnsw", false, false));     // No quantization, HNSW
    results.push_back(test_save_load("no_pq_exact", false, true));     // No quantization, exact k-NN
    results.push_back(test_save_load("with_pq_hnsw", true, false));    // With quantization, HNSW
    results.push_back(test_save_load("with_pq_exact", true, true));    // With quantization, exact k-NN

    printf("\n🏁 FINAL RESULTS\n");
    printf("================\n");

    int passed = 0;
    for (int i = 0; i < 4; i++) {
        const char* names[] = {"No PQ + HNSW", "No PQ + Exact", "PQ + HNSW", "PQ + Exact"};
        if (results[i]) {
            printf("✅ %s: PASSED\n", names[i]);
            passed++;
        } else {
            printf("❌ %s: FAILED\n", names[i]);
        }
    }

    printf("\nResult: %d/4 tests passed\n", passed);

    if (passed == 4) {
        printf("🎉 ALL TESTS PASSED! Save/load format fix is working!\n");
        return 0;
    } else {
        printf("💥 Some tests failed. Format issue still exists.\n");
        return 1;
    }
}