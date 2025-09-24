#include <iostream>
#include <vector>
#include <random>
#include <cstdio>
#include "uwot_simple_wrapper.h"

void test_hamming_validation() {
    printf("Testing Hamming metric validation...\n");

    // Create non-binary data (should trigger warning)
    const int n_obs = 100;
    const int n_dim = 10;
    const int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);
    std::vector<float> embedding(n_obs * embedding_dim);

    // Fill with non-binary data (random floats 0-5)
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 5.0f);

    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = dis(gen);
    }

    UwotModel* model = uwot_create_model();

    printf("Fitting with non-binary data using Hamming metric (should show warning):\n");
    int result = uwot_fit_with_progress(model, data.data(), n_obs, n_dim, embedding_dim,
                                       15, 0.1f, 1.0f, 50, UWOT_METRIC_HAMMING,
                                       embedding.data(), nullptr, 0, 16, 64, 32, 0);

    printf("Fit result: %s\n", (result == UWOT_SUCCESS) ? "SUCCESS" : "ERROR");

    uwot_destroy_model(model);
}

void test_correlation_validation() {
    printf("\nTesting Correlation metric validation...\n");

    // Create constant data (should trigger warning)
    const int n_obs = 100;
    const int n_dim = 10;
    const int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);
    std::vector<float> embedding(n_obs * embedding_dim);

    // Fill with constant data (no variance)
    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = 1.0f; // All same value
    }

    UwotModel* model = uwot_create_model();

    printf("Fitting with constant data using Correlation metric (should show warning):\n");
    int result = uwot_fit_with_progress(model, data.data(), n_obs, n_dim, embedding_dim,
                                       15, 0.1f, 1.0f, 50, UWOT_METRIC_CORRELATION,
                                       embedding.data(), nullptr, 0, 16, 64, 32, 0);

    printf("Fit result: %s\n", (result == UWOT_SUCCESS) ? "SUCCESS" : "ERROR");

    uwot_destroy_model(model);
}

void test_good_binary_data() {
    printf("\nTesting proper binary data with Hamming metric (should not show warning)...\n");

    // Create proper binary data
    const int n_obs = 100;
    const int n_dim = 10;
    const int embedding_dim = 2;

    std::vector<float> data(n_obs * n_dim);
    std::vector<float> embedding(n_obs * embedding_dim);

    // Fill with proper binary data (0 or 1)
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dis(0, 1);

    for (int i = 0; i < n_obs * n_dim; i++) {
        data[i] = static_cast<float>(dis(gen));
    }

    UwotModel* model = uwot_create_model();

    printf("Fitting with proper binary data using Hamming metric (no warning expected):\n");
    int result = uwot_fit_with_progress(model, data.data(), n_obs, n_dim, embedding_dim,
                                       15, 0.1f, 1.0f, 50, UWOT_METRIC_HAMMING,
                                       embedding.data(), nullptr, 0, 16, 64, 32, 0);

    printf("Fit result: %s\n", (result == UWOT_SUCCESS) ? "SUCCESS" : "ERROR");

    uwot_destroy_model(model);
}

int main() {
    printf("=== METRIC VALIDATION WARNING TESTS ===\n");
    printf("This test validates that inappropriate data triggers warnings for specific metrics.\n\n");

    test_hamming_validation();
    test_correlation_validation();
    test_good_binary_data();

    printf("\n=== Test completed ===\n");
    printf("Check the output above for expected warning messages.\n");

    return 0;
}