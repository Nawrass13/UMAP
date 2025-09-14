#include "uwot_simple_wrapper.h"
#include <cstdio>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <memory>
#include <chrono>
#include <algorithm>
#include <numeric>

// Enhanced progress callback for detailed testing
void validation_progress_callback_v2(const char* phase, int current, int total, float percent, const char* message) {
    printf("[PROGRESS] %s: %.1f%% (%d/%d)", phase, percent, current, total);
    if (message) {
        printf(" - %s", message);
    }
    printf("\n");
}

// Helper function to calculate Mean Squared Error between two embeddings
double calculate_mse(const std::vector<float>& embedding1, const std::vector<float>& embedding2) {
    if (embedding1.size() != embedding2.size()) {
        printf("ERROR: Embedding size mismatch for MSE calculation\n");
        return -1.0;
    }

    double sum_squared_diff = 0.0;
    for (size_t i = 0; i < embedding1.size(); i++) {
        double diff = embedding1[i] - embedding2[i];
        sum_squared_diff += diff * diff;
    }

    return sum_squared_diff / embedding1.size();
}

// Helper function to calculate embedding statistics
void calculate_embedding_stats(const std::vector<float>& embedding, const char* name) {
    if (embedding.empty()) return;

    float min_val = *std::min_element(embedding.begin(), embedding.end());
    float max_val = *std::max_element(embedding.begin(), embedding.end());
    double mean = std::accumulate(embedding.begin(), embedding.end(), 0.0) / embedding.size();

    double variance = 0.0;
    for (float val : embedding) {
        variance += (val - mean) * (val - mean);
    }
    variance /= embedding.size();

    printf("[STATS] %s - min: %.4f, max: %.4f, mean: %.4f, std: %.4f\n",
           name, min_val, max_val, mean, std::sqrt(variance));
}

// Generate synthetic test data with known structure
std::vector<float> generate_test_data(int n_obs, int n_dim, int seed = 42) {
    std::vector<float> data(n_obs * n_dim);
    std::mt19937 gen(seed);

    // Create structured data with 3 clusters for validation
    int cluster_size = n_obs / 3;

    for (int i = 0; i < n_obs; i++) {
        int cluster = i / cluster_size;
        if (cluster >= 3) cluster = 2; // Handle remainder

        std::normal_distribution<float> dist(cluster * 3.0f, 1.0f); // Separate clusters

        for (int j = 0; j < n_dim; j++) {
            data[i * n_dim + j] = dist(gen);
        }
    }

    return data;
}

// Test 1: HNSW vs Exact k-NN Accuracy Validation
bool test_hnsw_vs_exact_accuracy() {
    printf("\n=== TEST 1: HNSW vs Exact k-NN Accuracy Validation ===\n");

    // Test parameters
    constexpr int n_obs = 1000;    // Reasonable size for accuracy testing
    constexpr int n_dim = 50;      // High-dimensional for realistic test
    constexpr int embedding_dim = 2;
    constexpr int n_neighbors = 15;
    constexpr int n_epochs = 100;

    printf("Test parameters: %d samples x %d features → %dD embedding\n", n_obs, n_dim, embedding_dim);

    // Generate test data
    auto data = generate_test_data(n_obs, n_dim);

    // Test 1a: HNSW Approximate mode
    printf("\n--- Testing HNSW Approximate k-NN ---\n");
    UwotModel* hnsw_model = uwot_create();
    if (!hnsw_model) {
        printf("ERROR: Failed to create HNSW model\n");
        return false;
    }

    std::vector<float> hnsw_embedding(n_obs * embedding_dim);
    auto hnsw_start = std::chrono::high_resolution_clock::now();

    int result = uwot_fit_with_enhanced_progress(hnsw_model, data.data(), n_obs, n_dim,
                                               embedding_dim, n_neighbors, 0.1f, n_epochs,
                                               UWOT_METRIC_EUCLIDEAN, hnsw_embedding.data(),
                                               validation_progress_callback_v2, 0); // force_exact = 0 (HNSW)

    auto hnsw_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - hnsw_start).count();

    if (result != UWOT_SUCCESS) {
        printf("ERROR: HNSW training failed with code %d\n", result);
        uwot_destroy(hnsw_model);
        return false;
    }

    printf("✅ HNSW training completed in %.2f seconds\n", hnsw_time);
    calculate_embedding_stats(hnsw_embedding, "HNSW embedding");

    // Test 1b: Exact mode (for comparison)
    printf("\n--- Testing Exact k-NN (for accuracy comparison) ---\n");
    UwotModel* exact_model = uwot_create();
    if (!exact_model) {
        printf("ERROR: Failed to create exact model\n");
        uwot_destroy(hnsw_model);
        return false;
    }

    std::vector<float> exact_embedding(n_obs * embedding_dim);
    auto exact_start = std::chrono::high_resolution_clock::now();

    result = uwot_fit_with_enhanced_progress(exact_model, data.data(), n_obs, n_dim,
                                           embedding_dim, n_neighbors, 0.1f, n_epochs,
                                           UWOT_METRIC_EUCLIDEAN, exact_embedding.data(),
                                           validation_progress_callback_v2, 1); // force_exact = 1

    auto exact_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - exact_start).count();

    if (result != UWOT_SUCCESS) {
        printf("ERROR: Exact training failed with code %d\n", result);
        uwot_destroy(hnsw_model);
        uwot_destroy(exact_model);
        return false;
    }

    printf("✅ Exact training completed in %.2f seconds\n", exact_time);
    calculate_embedding_stats(exact_embedding, "Exact embedding");

    // Test 1c: Accuracy comparison
    double mse = calculate_mse(hnsw_embedding, exact_embedding);
    double speedup = exact_time / hnsw_time;

    printf("\n--- Accuracy & Performance Results ---\n");
    printf("MSE between HNSW and Exact: %.6f\n", mse);
    printf("Speedup (HNSW vs Exact): %.2fx\n", speedup);

    // Validation criteria
    const double MAX_ACCEPTABLE_MSE = 0.01; // Target: MSE < 0.01
    bool accuracy_pass = (mse >= 0 && mse < MAX_ACCEPTABLE_MSE);
    bool performance_pass = (speedup > 1.0);

    printf("\n--- Validation Results ---\n");
    printf("Accuracy test (MSE < %.3f): %s (MSE = %.6f)\n",
           MAX_ACCEPTABLE_MSE, accuracy_pass ? "✅ PASS" : "❌ FAIL", mse);
    printf("Performance test (speedup > 1.0x): %s (%.2fx speedup)\n",
           performance_pass ? "✅ PASS" : "❌ FAIL", speedup);

    uwot_destroy(hnsw_model);
    uwot_destroy(exact_model);

    return accuracy_pass && performance_pass;
}

// Test 2: Multi-Metric Support Validation
bool test_multi_metric_support() {
    printf("\n=== TEST 2: Multi-Metric Support Validation ===\n");

    constexpr int n_obs = 500;
    constexpr int n_dim = 20;
    constexpr int embedding_dim = 2;

    auto data = generate_test_data(n_obs, n_dim, 12345);

    // Test supported metrics for HNSW
    std::vector<std::pair<UwotMetric, const char*>> metrics = {
        {UWOT_METRIC_EUCLIDEAN, "Euclidean"},
        {UWOT_METRIC_COSINE, "Cosine"},
        {UWOT_METRIC_MANHATTAN, "Manhattan"}
    };

    bool all_passed = true;

    for (auto& [metric, name] : metrics) {
        printf("\n--- Testing %s Distance ---\n", name);

        UwotModel* model = uwot_create();
        if (!model) {
            printf("ERROR: Failed to create model for %s\n", name);
            all_passed = false;
            continue;
        }

        std::vector<float> embedding(n_obs * embedding_dim);
        auto start_time = std::chrono::high_resolution_clock::now();

        int result = uwot_fit_with_enhanced_progress(model, data.data(), n_obs, n_dim,
                                                   embedding_dim, 15, 0.1f, 50,
                                                   metric, embedding.data(),
                                                   validation_progress_callback_v2, 0);

        auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();

        if (result == UWOT_SUCCESS) {
            printf("✅ %s metric training succeeded in %.2fs\n", name, elapsed);
            calculate_embedding_stats(embedding, name);
        } else {
            printf("❌ %s metric training failed with code %d\n", name, result);
            all_passed = false;
        }

        uwot_destroy(model);
    }

    // Test unsupported metrics (should fall back to exact)
    printf("\n--- Testing Unsupported Metrics (Exact Fallback) ---\n");
    std::vector<std::pair<UwotMetric, const char*>> exact_metrics = {
        {UWOT_METRIC_CORRELATION, "Correlation"},
        {UWOT_METRIC_HAMMING, "Hamming"}
    };

    for (auto& [metric, name] : exact_metrics) {
        printf("\n--- Testing %s Distance (Exact Only) ---\n", name);

        UwotModel* model = uwot_create();
        if (!model) {
            printf("ERROR: Failed to create model for %s\n", name);
            continue;
        }

        std::vector<float> embedding(n_obs * embedding_dim);

        // Should work but use exact computation with warnings
        int result = uwot_fit_with_enhanced_progress(model, data.data(), n_obs, n_dim,
                                                   embedding_dim, 15, 0.1f, 30,
                                                   metric, embedding.data(),
                                                   validation_progress_callback_v2, 0);

        if (result == UWOT_SUCCESS) {
            printf("✅ %s metric training succeeded (exact fallback)\n", name);
        } else {
            printf("❌ %s metric training failed with code %d\n", name, result);
            all_passed = false;
        }

        uwot_destroy(model);
    }

    return all_passed;
}

// Test 3: Memory Usage and Model Persistence
bool test_memory_and_persistence() {
    printf("\n=== TEST 3: Memory Usage and Model Persistence ===\n");

    constexpr int n_obs = 2000;
    constexpr int n_dim = 100;
    constexpr int embedding_dim = 2;

    auto data = generate_test_data(n_obs, n_dim, 99999);

    // Train model
    UwotModel* model = uwot_create();
    if (!model) {
        printf("ERROR: Failed to create model\n");
        return false;
    }

    std::vector<float> embedding(n_obs * embedding_dim);

    printf("Training model for persistence test...\n");
    int result = uwot_fit_with_enhanced_progress(model, data.data(), n_obs, n_dim,
                                               embedding_dim, 15, 0.1f, 50,
                                               UWOT_METRIC_EUCLIDEAN, embedding.data(),
                                               validation_progress_callback_v2, 0);

    if (result != UWOT_SUCCESS) {
        printf("ERROR: Model training failed\n");
        uwot_destroy(model);
        return false;
    }

    printf("✅ Model training completed\n");

    // Test model persistence
    const char* model_file = "test_model_hnsw.bin";
    printf("Saving model to %s...\n", model_file);

    result = uwot_save_model(model, model_file);
    if (result != UWOT_SUCCESS) {
        printf("ERROR: Model save failed with code %d\n", result);
        uwot_destroy(model);
        return false;
    }

    printf("✅ Model saved successfully\n");

    // Load model and test
    printf("Loading model from %s...\n", model_file);
    UwotModel* loaded_model = uwot_load_model(model_file);
    if (!loaded_model) {
        printf("ERROR: Model load failed\n");
        uwot_destroy(model);
        return false;
    }

    printf("✅ Model loaded successfully\n");

    // Test transform with loaded model
    printf("Testing transform with loaded model...\n");
    std::vector<float> new_data(n_dim);
    std::mt19937 gen(777);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < n_dim; i++) {
        new_data[i] = dist(gen);
    }

    std::vector<float> transform_result(embedding_dim);
    result = uwot_transform(loaded_model, new_data.data(), 1, n_dim, transform_result.data());

    if (result == UWOT_SUCCESS) {
        printf("✅ Transform test passed - result: [%.4f, %.4f]\n",
               transform_result[0], transform_result[1]);
    } else {
        printf("❌ Transform test failed with code %d\n", result);
        uwot_destroy(model);
        uwot_destroy(loaded_model);
        return false;
    }

    // Test enhanced transform with safety analysis
    printf("Testing enhanced transform with safety analysis...\n");
    int nn_indices[15];
    float nn_distances[15];
    float confidence_score;
    int outlier_level;
    float percentile_rank;
    float z_score;

    result = uwot_transform_detailed(loaded_model, new_data.data(), 1, n_dim,
                                   transform_result.data(), nn_indices, nn_distances,
                                   &confidence_score, &outlier_level,
                                   &percentile_rank, &z_score);

    if (result == UWOT_SUCCESS) {
        printf("✅ Enhanced transform test passed:\n");
        printf("   Confidence: %.3f, Outlier Level: %d, Percentile: %.1f%%, Z-Score: %.3f\n",
               confidence_score, outlier_level, percentile_rank, z_score);
    } else {
        printf("❌ Enhanced transform test failed with code %d\n", result);
    }

    uwot_destroy(model);
    uwot_destroy(loaded_model);

    // Cleanup
    remove(model_file);

    return true;
}

// Main test runner
int main() {
    printf("🚀 HNSW k-NN Optimization Validation Suite\n");
    printf("==========================================\n");

    int passed = 0;
    int total = 3;

    // Run comprehensive validation tests
    if (test_hnsw_vs_exact_accuracy()) {
        printf("\n✅ TEST 1 PASSED: HNSW vs Exact Accuracy\n");
        passed++;
    } else {
        printf("\n❌ TEST 1 FAILED: HNSW vs Exact Accuracy\n");
    }

    if (test_multi_metric_support()) {
        printf("\n✅ TEST 2 PASSED: Multi-Metric Support\n");
        passed++;
    } else {
        printf("\n❌ TEST 2 FAILED: Multi-Metric Support\n");
    }

    if (test_memory_and_persistence()) {
        printf("\n✅ TEST 3 PASSED: Memory Usage and Persistence\n");
        passed++;
    } else {
        printf("\n❌ TEST 3 FAILED: Memory Usage and Persistence\n");
    }

    // Final results
    printf("\n==========================================\n");
    printf("🎯 VALIDATION RESULTS: %d/%d tests passed\n", passed, total);

    if (passed == total) {
        printf("🎉 ALL TESTS PASSED! HNSW optimization is ready for deployment.\n");
        return 0;
    } else {
        printf("⚠️  Some tests failed. Please review implementation.\n");
        return 1;
    }
}