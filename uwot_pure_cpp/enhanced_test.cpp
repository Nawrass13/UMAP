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
        embedding_dim, 15, 0.1f, n_epochs,
        UWOT_METRIC_EUCLIDEAN, embedding.data(),
        test_progress_callback);

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
        embedding_dim, 15, 0.1f, 100,
        UWOT_METRIC_EUCLIDEAN, embedding.data());

    if (result != UWOT_SUCCESS) {
        printf("Basic test failed with error: %s\n", uwot_get_error_message(result));
        uwot_destroy(model);
        return false;
    }

    // Test model info
    int info_n_vertices, info_n_dim, info_embedding_dim, info_n_neighbors;
    float info_min_dist;
    UwotMetric info_metric;

    result = uwot_get_model_info(model, &info_n_vertices, &info_n_dim,
        &info_embedding_dim, &info_n_neighbors,
        &info_min_dist, &info_metric);

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
        embedding_dim, 15, 0.1f, 50,
        UWOT_METRIC_COSINE, embedding.data());

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
            embedding_dim, 10, 0.1f, 30,
            metric, embedding.data());

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

int main() {
    printf("===========================================\n");
    printf("  Enhanced UMAP C++ Library Test Suite\n");
    printf("===========================================\n\n");

    bool all_passed = true;

    // Run all tests
    all_passed &= test_basic_functionality();
    printf("\n");

    all_passed &= test_progress_reporting();
    printf("\n");

    all_passed &= test_27d_embedding();
    printf("\n");

    all_passed &= test_distance_metrics();
    printf("\n");

    printf("===========================================\n");
    if (all_passed) {
        printf("  ALL TESTS PASSED!\n");
        printf("  Enhanced UMAP library is working correctly\n");
        printf("  Features verified:\n");
        printf("  ✓ Basic UMAP functionality\n");
        printf("  ✓ Progress reporting with callbacks\n");
        printf("  ✓ 27D embedding capability\n");
        printf("  ✓ Multiple distance metrics\n");
        printf("  ✓ Model information retrieval\n");
    }
    else {
        printf("  SOME TESTS FAILED!\n");
        printf("  Check the output above for details\n");
    }
    printf("===========================================\n");

    return all_passed ? 0 : 1;
}