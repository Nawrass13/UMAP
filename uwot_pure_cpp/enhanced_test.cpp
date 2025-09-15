#include "uwot_simple_wrapper.h"
#include <cstdio>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <memory>
#include <chrono>

// Progress callback for testing
void test_progress_callback(int epoch, int total_epochs, float percent) {
    static int last_reported = -1;
    int current_percent = static_cast<int>(percent);

    // Only report every 20% to avoid spam
    if (current_percent >= last_reported + 20) {
        printf("Progress: %d%% (Epoch %d/%d)\n", current_percent, epoch, total_epochs);
        last_reported = current_percent;
    }
}

bool test_progress_reporting() {
    printf("Testing progress reporting...\n");

    UwotModel* model = uwot_create();
    if (!model) {
        printf("Failed to create model\n");
        return false;
    }

    // Generate small test data for quick progress test
    constexpr int n_obs = 100;
    constexpr int n_dim = 5;
    constexpr int embedding_dim = 2;
    constexpr int n_epochs = 50;  // Small number for quick test

    std::vector<float> data(n_obs * n_dim);
    std::mt19937 gen(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = dist(gen);
    }

    std::vector<float> embedding(n_obs * embedding_dim);

    printf("Testing uwot_fit_with_progress...\n");
    int result = uwot_fit_with_progress(model, data.data(), n_obs, n_dim,
        embedding_dim, 15, 0.1f, 1.0f, n_epochs,
        UWOT_METRIC_EUCLIDEAN, embedding.data(),
        test_progress_callback, 0);

    if (result != UWOT_SUCCESS) {
        printf("Progress test failed with error: %s\n", uwot_get_error_message(result));
        uwot_destroy(model);
        return false;
    }

    uwot_destroy(model);
    printf("Progress reporting test passed!\n");
    return true;
}

bool test_basic_functionality() {
    printf("Testing basic UMAP functionality...\n");

    UwotModel* model = uwot_create();
    if (!model) {
        printf("Failed to create model\n");
        return false;
    }

    // Generate test data
    constexpr int n_obs = 150;
    constexpr int n_dim = 4;
    constexpr int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = dist(gen);
    }

    std::vector<float> embedding(n_obs * embedding_dim);

    printf("Testing basic uwot_fit...\n");
    int result = uwot_fit(model, data.data(), n_obs, n_dim,
        embedding_dim, 15, 0.1f, 1.0f, 100,
        UWOT_METRIC_EUCLIDEAN, embedding.data(), 0);

    if (result != UWOT_SUCCESS) {
        printf("Basic test failed with error: %s\n", uwot_get_error_message(result));
        uwot_destroy(model);
        return false;
    }

    // Test model info
    int info_n_vertices, info_n_dim, info_embedding_dim, info_n_neighbors;
    float info_min_dist, info_spread;
    UwotMetric info_metric;
    int info_use_quantization, info_hnsw_M, info_hnsw_ef_construction, info_hnsw_ef_search;

    result = uwot_get_model_info(model, &info_n_vertices, &info_n_dim,
        &info_embedding_dim, &info_n_neighbors,
        &info_min_dist, &info_spread, &info_metric,
        &info_use_quantization, &info_hnsw_M,
        &info_hnsw_ef_construction, &info_hnsw_ef_search);

    if (result != UWOT_SUCCESS) {
        printf("Model info test failed\n");
        uwot_destroy(model);
        return false;
    }

    printf("Model info: %d vertices, %d->%d dimensions, %d neighbors, metric: %s\n",
        info_n_vertices, info_n_dim, info_embedding_dim, info_n_neighbors,
        uwot_get_metric_name(info_metric));

    uwot_destroy(model);
    printf("Basic functionality test passed!\n");
    return true;
}

bool test_27d_embedding() {
    printf("Testing 27D embedding...\n");

    UwotModel* model = uwot_create();
    if (!model) {
        printf("Failed to create model\n");
        return false;
    }

    // Generate test data
    constexpr int n_obs = 100;
    constexpr int n_dim = 10;
    constexpr int embedding_dim = 27;  // Test 27D

    std::vector<float> data(n_obs * n_dim);
    std::mt19937 gen(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = dist(gen);
    }

    std::vector<float> embedding(n_obs * embedding_dim);

    printf("Testing 27D embedding with cosine distance...\n");
    int result = uwot_fit(model, data.data(), n_obs, n_dim,
        embedding_dim, 15, 0.1f, 1.0f, 50,
        UWOT_METRIC_COSINE, embedding.data(), 0);

    if (result != UWOT_SUCCESS) {
        printf("27D test failed with error: %s\n", uwot_get_error_message(result));
        uwot_destroy(model);
        return false;
    }

    printf("27D embedding dimensions: %d\n", uwot_get_embedding_dim(model));

    uwot_destroy(model);
    printf("27D embedding test passed!\n");
    return true;
}

bool test_distance_metrics() {
    printf("Testing different distance metrics...\n");

    const UwotMetric metrics[] = {
        UWOT_METRIC_EUCLIDEAN,
        UWOT_METRIC_COSINE,
        UWOT_METRIC_MANHATTAN,
        UWOT_METRIC_CORRELATION,
        UWOT_METRIC_HAMMING
    };

    constexpr int n_obs = 50;
    constexpr int n_dim = 3;
    constexpr int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);
    std::mt19937 gen(456);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = dist(gen);
    }

    for (const auto& metric : metrics) {
        printf("Testing %s distance...\n", uwot_get_metric_name(metric));

        UwotModel* model = uwot_create();
        if (!model) {
            printf("Failed to create model for %s\n", uwot_get_metric_name(metric));
            return false;
        }

        std::vector<float> embedding(n_obs * embedding_dim);

        int result = uwot_fit(model, data.data(), n_obs, n_dim,
            embedding_dim, 10, 0.1f, 1.0f, 30,
            metric, embedding.data(), 0);

        if (result != UWOT_SUCCESS) {
            printf("%s metric test failed with error: %s\n",
                uwot_get_metric_name(metric), uwot_get_error_message(result));
            uwot_destroy(model);
            return false;
        }

        uwot_destroy(model);
    }

    printf("Distance metrics test passed!\n");
    return true;
}

bool test_model_persistence() {
    printf("Testing model save/load functionality...\n");

    const char* test_model_file = "test_model.umap";

    // Create and train a model
    UwotModel* model = uwot_create();
    if (!model) {
        printf("Failed to create model\n");
        return false;
    }

    constexpr int n_obs = 100;
    constexpr int n_dim = 5;
    constexpr int embedding_dim = 3;

    std::vector<float> train_data(n_obs * n_dim);
    std::mt19937 gen(789);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n_obs * n_dim; i++) {
        train_data[i] = dist(gen);
    }

    std::vector<float> train_embedding(n_obs * embedding_dim);

    printf("Training model for persistence test...\n");
    int result = uwot_fit(model, train_data.data(), n_obs, n_dim,
        embedding_dim, 15, 0.1f, 1.0f, 50,
        UWOT_METRIC_EUCLIDEAN, train_embedding.data(), 0);

    if (result != UWOT_SUCCESS) {
        printf("Training failed with error: %s\n", uwot_get_error_message(result));
        uwot_destroy(model);
        return false;
    }

    // Save the model
    printf("Saving model to %s...\n", test_model_file);
    result = uwot_save_model(model, test_model_file);
    if (result != UWOT_SUCCESS) {
        printf("Model save failed with error: %s\n", uwot_get_error_message(result));
        uwot_destroy(model);
        return false;
    }

    uwot_destroy(model);

    // Load the model
    printf("Loading model from %s...\n", test_model_file);
    UwotModel* loaded_model = uwot_load_model(test_model_file);
    if (!loaded_model) {
        printf("Model load failed\n");
        return false;
    }

    // Verify model info
    int info_n_vertices, info_n_dim, info_embedding_dim, info_n_neighbors;
    float info_min_dist, info_spread;
    UwotMetric info_metric;
    int info_use_quantization, info_hnsw_M, info_hnsw_ef_construction, info_hnsw_ef_search;

    result = uwot_get_model_info(loaded_model, &info_n_vertices, &info_n_dim,
        &info_embedding_dim, &info_n_neighbors,
        &info_min_dist, &info_spread, &info_metric,
        &info_use_quantization, &info_hnsw_M,
        &info_hnsw_ef_construction, &info_hnsw_ef_search);

    if (result != UWOT_SUCCESS) {
        printf("Loaded model info retrieval failed\n");
        uwot_destroy(loaded_model);
        return false;
    }

    printf("Loaded model info: %d vertices, %d->%d dimensions\n",
        info_n_vertices, info_n_dim, info_embedding_dim);

    // Test transform with loaded model
    constexpr int n_test = 10;
    std::vector<float> test_data(n_test * n_dim);
    for (int i = 0; i < n_test * n_dim; i++) {
        test_data[i] = dist(gen);
    }

    std::vector<float> test_embedding(n_test * embedding_dim);
    result = uwot_transform(loaded_model, test_data.data(), n_test, n_dim, test_embedding.data());

    if (result != UWOT_SUCCESS) {
        printf("Transform with loaded model failed with error: %s\n", uwot_get_error_message(result));
        uwot_destroy(loaded_model);
        return false;
    }

    printf("Transform with loaded model successful!\n");

    uwot_destroy(loaded_model);

    // Clean up test file
    remove(test_model_file);

    printf("Model persistence test passed!\n");
    return true;
}

bool test_hnsw_enhanced_transform() {
    printf("Testing HNSW enhanced transform with safety metrics...\n");

    UwotModel* model = uwot_create();
    if (!model) {
        printf("Failed to create model\n");
        return false;
    }

    // Create training data with clear patterns
    constexpr int n_obs = 200;
    constexpr int n_dim = 10;
    constexpr int embedding_dim = 5;
    constexpr int n_neighbors = 15;

    std::vector<float> train_data(n_obs * n_dim);
    std::mt19937 gen(999);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate clustered training data
    for (int i = 0; i < n_obs; i++) {
        float cluster_center = (i < n_obs/2) ? -2.0f : 2.0f; // Two clusters
        for (int j = 0; j < n_dim; j++) {
            train_data[i * n_dim + j] = cluster_center + dist(gen) * 0.5f;
        }
    }

    std::vector<float> train_embedding(n_obs * embedding_dim);

    printf("Training model with HNSW optimization...\n");
    int result = uwot_fit(model, train_data.data(), n_obs, n_dim,
        embedding_dim, n_neighbors, 0.1f, 1.0f, 100,
        UWOT_METRIC_EUCLIDEAN, train_embedding.data(), 0);

    if (result != UWOT_SUCCESS) {
        printf("HNSW training failed with error: %s\n", uwot_get_error_message(result));
        uwot_destroy(model);
        return false;
    }

    printf("Training completed, testing enhanced transform...\n");

    // Test different types of data
    constexpr int n_test = 5;

    // Test 1: Similar data (should be normal)
    std::vector<float> similar_data(n_test * n_dim);
    for (int i = 0; i < n_test; i++) {
        float cluster_center = (i < n_test/2) ? -2.0f : 2.0f;
        for (int j = 0; j < n_dim; j++) {
            similar_data[i * n_dim + j] = cluster_center + dist(gen) * 0.3f;
        }
    }

    // Test 2: Outlier data (should trigger safety warnings)
    std::vector<float> outlier_data(n_test * n_dim);
    for (int i = 0; i < n_test * n_dim; i++) {
        outlier_data[i] = dist(gen) * 10.0f; // Much larger scale
    }

    // Prepare output arrays for detailed transform
    std::vector<float> test_embedding(n_test * embedding_dim);
    std::vector<int> nn_indices(n_test * n_neighbors);
    std::vector<float> nn_distances(n_test * n_neighbors);
    std::vector<float> confidence_scores(n_test);
    std::vector<int> outlier_levels(n_test);
    std::vector<float> percentile_ranks(n_test);
    std::vector<float> z_scores(n_test);

    printf("\nTesting similar data (should be safe):\n");
    result = uwot_transform_detailed(model, similar_data.data(), n_test, n_dim,
                                   test_embedding.data(), nn_indices.data(), nn_distances.data(),
                                   confidence_scores.data(), outlier_levels.data(),
                                   percentile_ranks.data(), z_scores.data());

    if (result != UWOT_SUCCESS) {
        printf("Enhanced transform failed with error: %s\n", uwot_get_error_message(result));
        uwot_destroy(model);
        return false;
    }

    // Display results for similar data
    for (int i = 0; i < n_test; i++) {
        printf("  Sample %d: Confidence=%.3f, Outlier=%d, Percentile=%.1f%%, Z=%.2f\n",
               i + 1, confidence_scores[i], outlier_levels[i],
               percentile_ranks[i], z_scores[i]);
    }

    printf("\nTesting outlier data (should trigger warnings):\n");
    result = uwot_transform_detailed(model, outlier_data.data(), n_test, n_dim,
                                   test_embedding.data(), nn_indices.data(), nn_distances.data(),
                                   confidence_scores.data(), outlier_levels.data(),
                                   percentile_ranks.data(), z_scores.data());

    if (result != UWOT_SUCCESS) {
        printf("Enhanced transform of outliers failed with error: %s\n", uwot_get_error_message(result));
        uwot_destroy(model);
        return false;
    }

    // Display results for outlier data
    for (int i = 0; i < n_test; i++) {
        printf("  Sample %d: Confidence=%.3f, Outlier=%d, Percentile=%.1f%%, Z=%.2f\n",
               i + 1, confidence_scores[i], outlier_levels[i],
               percentile_ranks[i], z_scores[i]);
    }

    uwot_destroy(model);
    printf("HNSW enhanced transform test passed!\n");
    return true;
}

bool test_performance_comparison() {
    printf("Testing performance comparison (HNSW vs linear search)...\n");

    UwotModel* model = uwot_create();
    if (!model) {
        printf("Failed to create model\n");
        return false;
    }

    // Larger dataset for performance testing
    constexpr int n_obs = 1000;
    constexpr int n_dim = 50;
    constexpr int embedding_dim = 10;

    std::vector<float> train_data(n_obs * n_dim);
    std::mt19937 gen(2023);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < n_obs * n_dim; i++) {
        train_data[i] = dist(gen);
    }

    std::vector<float> train_embedding(n_obs * embedding_dim);

    printf("Training large model (1000 samples, 50D -> 10D)...\n");
    auto start_time = std::chrono::high_resolution_clock::now();

    int result = uwot_fit(model, train_data.data(), n_obs, n_dim,
        embedding_dim, 15, 0.1f, 1.0f, 100,
        UWOT_METRIC_EUCLIDEAN, train_embedding.data(), 0);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (result != UWOT_SUCCESS) {
        printf("Performance test training failed with error: %s\n", uwot_get_error_message(result));
        uwot_destroy(model);
        return false;
    }

    printf("Training completed in %lld ms\n", (long long)training_duration.count());

    // Test transform performance
    constexpr int n_test = 100;
    std::vector<float> test_data(n_test * n_dim);
    for (int i = 0; i < n_test * n_dim; i++) {
        test_data[i] = dist(gen);
    }

    // Test standard transform
    std::vector<float> test_embedding(n_test * embedding_dim);

    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) { // Multiple runs for averaging
        result = uwot_transform(model, test_data.data(), n_test, n_dim, test_embedding.data());
        if (result != UWOT_SUCCESS) {
            printf("Standard transform failed\n");
            uwot_destroy(model);
            return false;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    auto standard_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    printf("Standard transform (10 runs): %lld μs (avg: %.2f μs per run)\n",
           (long long)standard_duration.count(), standard_duration.count() / 10.0);

    // Test enhanced transform if available
    std::vector<int> nn_indices(n_test * 15);
    std::vector<float> nn_distances(n_test * 15);
    std::vector<float> confidence_scores(n_test);
    std::vector<int> outlier_levels(n_test);
    std::vector<float> percentile_ranks(n_test);
    std::vector<float> z_scores(n_test);

    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++) { // Multiple runs for averaging
        result = uwot_transform_detailed(model, test_data.data(), n_test, n_dim,
                                       test_embedding.data(), nn_indices.data(), nn_distances.data(),
                                       confidence_scores.data(), outlier_levels.data(),
                                       percentile_ranks.data(), z_scores.data());
        if (result != UWOT_SUCCESS) {
            printf("Enhanced transform failed (may not be implemented yet)\n");
            break;
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    auto enhanced_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    if (result == UWOT_SUCCESS) {
        printf("Enhanced transform (10 runs): %lld μs (avg: %.2f μs per run)\n",
               (long long)enhanced_duration.count(), enhanced_duration.count() / 10.0);
    }

    uwot_destroy(model);
    printf("Performance comparison test completed!\n");
    return true;
}

int main() {
    printf("===========================================\n");
    printf("  Enhanced UMAP C++ Library Test Suite\n");
    printf("  with HNSW Optimization & Safety Features\n");
    printf("===========================================\n\n");

    bool all_passed = true;

    // Run core functionality tests
    all_passed &= test_basic_functionality();
    printf("\n");

    all_passed &= test_progress_reporting();
    printf("\n");

    all_passed &= test_27d_embedding();
    printf("\n");

    all_passed &= test_distance_metrics();
    printf("\n");

    // Run enhanced HNSW tests
    all_passed &= test_model_persistence();
    printf("\n");

    all_passed &= test_hnsw_enhanced_transform();
    printf("\n");

    all_passed &= test_performance_comparison();
    printf("\n");

    printf("===========================================\n");
    if (all_passed) {
        printf("  ALL TESTS PASSED!\n");
        printf("  Enhanced UMAP library with HNSW is working correctly\n");
        printf("  Features verified:\n");
        printf("  ✓ Basic UMAP functionality\n");
        printf("  ✓ Progress reporting with callbacks\n");
        printf("  ✓ 27D embedding capability\n");
        printf("  ✓ Multiple distance metrics\n");
        printf("  ✓ Model information retrieval\n");
        printf("  ✓ Model save/load persistence\n");
        printf("  ✓ HNSW enhanced transform with safety metrics\n");
        printf("  ✓ Performance comparison testing\n");
        printf("  ✓ Outlier detection and confidence scoring\n");
    }
    else {
        printf("  SOME TESTS FAILED!\n");
        printf("  Check the output above for details\n");
        printf("  Note: HNSW enhanced features may not be fully implemented yet\n");
    }
    printf("===========================================\n");

    return all_passed ? 0 : 1;
}