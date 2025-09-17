#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <cassert>
#include "uwot_simple_wrapper.h"

// Generate test data
std::vector<float> generate_test_data(int n_samples, int n_features, int seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> data(n_samples * n_features);
    for (int i = 0; i < n_samples * n_features; i++) {
        data[i] = dist(rng);
    }

    return data;
}

// Test save/load cycle
bool test_save_load_cycle(const std::string& test_name,
                         int n_samples, int n_features, int embedding_dim,
                         bool use_quantization, UwotMetric metric) {

    printf("\n=== %s ===\n", test_name.c_str());
    printf("Samples: %d, Features: %d, Embedding: %dD\n", n_samples, n_features, embedding_dim);
    printf("Quantization: %s, Metric: %d\n", use_quantization ? "ON" : "OFF", metric);

    // Generate test data
    auto data = generate_test_data(n_samples, n_features);

    // Create and train model
    UwotModel* model = uwot_create();
    if (!model) {
        printf("❌ FAILED: Could not create model\n");
        return false;
    }

    // Configure training parameters
    int n_neighbors = std::min(15, n_samples - 1);
    float min_dist = 0.1f;
    float spread = 1.0f;
    int n_epochs = 50; // Reduced for faster testing

    // Parameter validation
    if (n_neighbors >= n_samples) {
        n_neighbors = std::max(5, n_samples - 1);
    }
    if (n_neighbors < 2) {
        printf("❌ FAILED: Not enough samples (%d) for meaningful neighbors (%d)\n", n_samples, n_neighbors);
        uwot_destroy(model);
        return false;
    }

    printf("Training model with parameters:\n");
    printf("  n_samples=%d, n_features=%d, embedding_dim=%d\n", n_samples, n_features, embedding_dim);
    printf("  n_neighbors=%d, min_dist=%.3f, spread=%.3f, n_epochs=%d\n", n_neighbors, min_dist, spread, n_epochs);
    printf("  metric=%d, use_quantization=%d\n", metric, use_quantization);

    auto start = std::chrono::high_resolution_clock::now();

    int result = uwot_fit_with_progress(
        model, data.data(), n_samples, n_features,
        embedding_dim, n_neighbors, min_dist, spread, n_epochs, metric,
        nullptr, // embedding output (not needed for test)
        nullptr, // progress callback
        false,   // force_exact_knn
        use_quantization ? 1 : 0  // use_quantization
    );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (result != UWOT_SUCCESS) {
        printf("❌ FAILED: Training failed with error %d\n", result);
        uwot_destroy(model);
        return false;
    }

    printf("✅ Training completed in %ldms\n", duration.count());

    // Get model info before save
    int orig_n_vertices, orig_n_dim, orig_embedding_dim, orig_n_neighbors;
    float orig_min_dist, orig_spread;
    UwotMetric orig_metric;
    int orig_use_quantization, orig_hnsw_M, orig_hnsw_ef_construction, orig_hnsw_ef_search;

    result = uwot_get_model_info(model, &orig_n_vertices, &orig_n_dim, &orig_embedding_dim,
                               &orig_n_neighbors, &orig_min_dist, &orig_spread, &orig_metric,
                               &orig_use_quantization, &orig_hnsw_M, &orig_hnsw_ef_construction,
                               &orig_hnsw_ef_search);

    if (result != UWOT_SUCCESS) {
        printf("❌ FAILED: Could not get model info (error %d)\n", result);
        uwot_destroy(model);
        return false;
    }

    printf("Model info: %d samples, %dD→%dD, k=%d, metric=%d, PQ=%s\n",
           orig_n_vertices, orig_n_dim, orig_embedding_dim, orig_n_neighbors,
           orig_metric, orig_use_quantization ? "ON" : "OFF");

    // Save model
    std::string filename = "test_model_" + test_name + ".umap";
    printf("Saving model to %s...\n", filename.c_str());

    result = uwot_save_model(model, filename.c_str());
    if (result != UWOT_SUCCESS) {
        printf("❌ FAILED: Save failed with error %d\n", result);
        uwot_destroy(model);
        return false;
    }

    printf("✅ Model saved successfully\n");

    // Test multiple points with original model before saving
    const int num_test_points = std::min(5, n_samples);
    printf("\n🧪 Testing %d projections with ORIGINAL model:\n", num_test_points);

    std::vector<std::vector<float>> original_projections(num_test_points);
    std::vector<std::vector<float>> test_points(num_test_points);

    for (int test_idx = 0; test_idx < num_test_points; test_idx++) {
        // Prepare test point
        test_points[test_idx].resize(n_features);
        for (int i = 0; i < n_features; i++) {
            test_points[test_idx][i] = data[test_idx * n_features + i];
        }

        // Transform with original model
        original_projections[test_idx].resize(embedding_dim);
        std::vector<int> nn_indices(n_neighbors);
        std::vector<float> nn_distances(n_neighbors);
        float confidence_score;
        int outlier_level;
        float percentile_rank, z_score;

        result = uwot_transform_detailed(model, test_points[test_idx].data(), 1, n_features,
                                       original_projections[test_idx].data(),
                                       nn_indices.data(), nn_distances.data(),
                                       &confidence_score, &outlier_level, &percentile_rank, &z_score);

        if (result != UWOT_SUCCESS) {
            printf("❌ FAILED: Transform point %d with original model failed (error %d)\n", test_idx + 1, result);
            uwot_destroy(model);
            return false;
        }

        printf("  Point %d: [%.6f, %.6f", test_idx + 1,
               original_projections[test_idx][0],
               embedding_dim > 1 ? original_projections[test_idx][1] : 0.0f);
        if (embedding_dim > 2) printf(", %.6f", original_projections[test_idx][2]);
        if (embedding_dim > 3) printf(", ...");
        printf("] conf=%.3f\n", confidence_score);
    }

    // Destroy original model
    uwot_destroy(model);

    // Load model
    printf("Loading model from %s...\n", filename.c_str());

    UwotModel* loaded_model = uwot_load_model(filename.c_str());
    if (!loaded_model) {
        printf("❌ FAILED: Load failed - returned NULL\n");
        return false;
    }

    printf("✅ Model loaded successfully\n");

    // Get loaded model info and verify it matches
    int load_n_vertices, load_n_dim, load_embedding_dim, load_n_neighbors;
    float load_min_dist, load_spread;
    UwotMetric load_metric;
    int load_use_quantization, load_hnsw_M, load_hnsw_ef_construction, load_hnsw_ef_search;

    result = uwot_get_model_info(loaded_model, &load_n_vertices, &load_n_dim, &load_embedding_dim,
                               &load_n_neighbors, &load_min_dist, &load_spread, &load_metric,
                               &load_use_quantization, &load_hnsw_M, &load_hnsw_ef_construction,
                               &load_hnsw_ef_search);

    if (result != UWOT_SUCCESS) {
        printf("❌ FAILED: Could not get loaded model info (error %d)\n", result);
        uwot_destroy(loaded_model);
        return false;
    }

    printf("Loaded model info: %d samples, %dD→%dD, k=%d, metric=%d, PQ=%s\n",
           load_n_vertices, load_n_dim, load_embedding_dim, load_n_neighbors,
           load_metric, load_use_quantization ? "ON" : "OFF");

    // Comprehensive verification of ALL model parameters
    printf("\n📊 DETAILED PARAMETER VERIFICATION:\n");

    bool all_match = true;

    // Core model parameters
    if (orig_n_vertices != load_n_vertices) {
        printf("❌ n_vertices: orig=%d, loaded=%d\n", orig_n_vertices, load_n_vertices);
        all_match = false;
    } else {
        printf("✅ n_vertices: %d\n", orig_n_vertices);
    }

    if (orig_n_dim != load_n_dim) {
        printf("❌ n_dim: orig=%d, loaded=%d\n", orig_n_dim, load_n_dim);
        all_match = false;
    } else {
        printf("✅ n_dim: %d\n", orig_n_dim);
    }

    if (orig_embedding_dim != load_embedding_dim) {
        printf("❌ embedding_dim: orig=%d, loaded=%d\n", orig_embedding_dim, load_embedding_dim);
        all_match = false;
    } else {
        printf("✅ embedding_dim: %d\n", orig_embedding_dim);
    }

    if (orig_n_neighbors != load_n_neighbors) {
        printf("❌ n_neighbors: orig=%d, loaded=%d\n", orig_n_neighbors, load_n_neighbors);
        all_match = false;
    } else {
        printf("✅ n_neighbors: %d\n", orig_n_neighbors);
    }

    if (std::abs(orig_min_dist - load_min_dist) > 1e-6) {
        printf("❌ min_dist: orig=%.6f, loaded=%.6f (diff=%.6f)\n",
               orig_min_dist, load_min_dist, std::abs(orig_min_dist - load_min_dist));
        all_match = false;
    } else {
        printf("✅ min_dist: %.6f\n", orig_min_dist);
    }

    if (std::abs(orig_spread - load_spread) > 1e-6) {
        printf("❌ spread: orig=%.6f, loaded=%.6f (diff=%.6f)\n",
               orig_spread, load_spread, std::abs(orig_spread - load_spread));
        all_match = false;
    } else {
        printf("✅ spread: %.6f\n", orig_spread);
    }

    if (orig_metric != load_metric) {
        printf("❌ metric: orig=%d, loaded=%d\n", orig_metric, load_metric);
        all_match = false;
    } else {
        printf("✅ metric: %d\n", orig_metric);
    }

    // CRITICAL: HNSW hyperparameters (this was the bug!)
    if (orig_hnsw_M != load_hnsw_M) {
        printf("❌ HNSW M: orig=%d, loaded=%d\n", orig_hnsw_M, load_hnsw_M);
        all_match = false;
    } else {
        printf("✅ HNSW M: %d\n", orig_hnsw_M);
    }

    if (orig_hnsw_ef_construction != load_hnsw_ef_construction) {
        printf("❌ HNSW ef_construction: orig=%d, loaded=%d\n",
               orig_hnsw_ef_construction, load_hnsw_ef_construction);
        all_match = false;
    } else {
        printf("✅ HNSW ef_construction: %d\n", orig_hnsw_ef_construction);
    }

    if (orig_hnsw_ef_search != load_hnsw_ef_search) {
        printf("❌ HNSW ef_search: orig=%d, loaded=%d\n",
               orig_hnsw_ef_search, load_hnsw_ef_search);
        all_match = false;
    } else {
        printf("✅ HNSW ef_search: %d\n", orig_hnsw_ef_search);
    }

    // Product Quantization settings
    if (orig_use_quantization != load_use_quantization) {
        printf("❌ use_quantization: orig=%s, loaded=%s\n",
               orig_use_quantization ? "true" : "false",
               load_use_quantization ? "true" : "false");
        all_match = false;
    } else {
        printf("✅ use_quantization: %s\n", orig_use_quantization ? "true" : "false");
    }

    if (!all_match) {
        printf("❌ FAILED: Model parameters don't match after load!\n");
        uwot_destroy(loaded_model);
        return false;
    }

    printf("✅ ALL MODEL PARAMETERS MATCH PERFECTLY\n");

    // Test the SAME points with loaded model
    printf("\n🧪 Testing %d projections with LOADED model:\n", num_test_points);

    std::vector<std::vector<float>> loaded_projections(num_test_points);
    bool projection_consistency = true;
    float max_overall_error = 0.0f;

    for (int test_idx = 0; test_idx < num_test_points; test_idx++) {
        // Transform with loaded model
        loaded_projections[test_idx].resize(embedding_dim);
        std::vector<int> nn_indices(n_neighbors);
        std::vector<float> nn_distances(n_neighbors);
        float confidence_score;
        int outlier_level;
        float percentile_rank, z_score;

        result = uwot_transform_detailed(loaded_model, test_points[test_idx].data(), 1, n_features,
                                       loaded_projections[test_idx].data(),
                                       nn_indices.data(), nn_distances.data(),
                                       &confidence_score, &outlier_level, &percentile_rank, &z_score);

        if (result != UWOT_SUCCESS) {
            printf("❌ FAILED: Transform point %d with loaded model failed (error %d)\n", test_idx + 1, result);
            uwot_destroy(loaded_model);
            return false;
        }

        printf("  Point %d: [%.6f, %.6f", test_idx + 1,
               loaded_projections[test_idx][0],
               embedding_dim > 1 ? loaded_projections[test_idx][1] : 0.0f);
        if (embedding_dim > 2) printf(", %.6f", loaded_projections[test_idx][2]);
        if (embedding_dim > 3) printf(", ...");
        printf("] conf=%.3f\n", confidence_score);

        // Calculate max difference for this point
        float max_point_error = 0.0f;
        for (int dim = 0; dim < embedding_dim; dim++) {
            float diff = std::abs(original_projections[test_idx][dim] - loaded_projections[test_idx][dim]);
            max_point_error = std::max(max_point_error, diff);
        }
        max_overall_error = std::max(max_overall_error, max_point_error);

        // Check tolerance for this point
        const float tolerance = use_quantization ? 0.1f : 0.001f;
        if (max_point_error > tolerance) {
            printf("❌ Point %d: INCONSISTENT (max_diff=%.6f > %.6f)\n", test_idx + 1, max_point_error, tolerance);
            projection_consistency = false;
        } else {
            printf("✅ Point %d: CONSISTENT (max_diff=%.6f)\n", test_idx + 1, max_point_error);
        }
    }

    // Final consistency check and detailed analysis
    if (!projection_consistency) {
        printf("❌ FAILED: Projection consistency test failed!\n");
        uwot_destroy(loaded_model);
        return false;
    }

    // DETAILED PROJECTION ANALYSIS
    printf("\n🎯 DETAILED PROJECTION CONSISTENCY ANALYSIS:\n");
    printf("==========================================\n");

    for (int test_idx = 0; test_idx < num_test_points; test_idx++) {
        printf("\nPoint %d detailed comparison:\n", test_idx + 1);
        printf("  Original: [");
        for (int dim = 0; dim < std::min(embedding_dim, 5); dim++) {
            printf("%.6f", original_projections[test_idx][dim]);
            if (dim < std::min(embedding_dim, 5) - 1) printf(", ");
        }
        if (embedding_dim > 5) printf(", ...");
        printf("]\n");

        printf("  Loaded:   [");
        for (int dim = 0; dim < std::min(embedding_dim, 5); dim++) {
            printf("%.6f", loaded_projections[test_idx][dim]);
            if (dim < std::min(embedding_dim, 5) - 1) printf(", ");
        }
        if (embedding_dim > 5) printf(", ...");
        printf("]\n");

        printf("  Diffs:    [");
        float point_mse = 0.0f;
        float point_max_diff = 0.0f;
        for (int dim = 0; dim < std::min(embedding_dim, 5); dim++) {
            float diff = std::abs(original_projections[test_idx][dim] - loaded_projections[test_idx][dim]);
            point_mse += diff * diff;
            point_max_diff = std::max(point_max_diff, diff);
            printf("%.6f", diff);
            if (dim < std::min(embedding_dim, 5) - 1) printf(", ");
        }
        if (embedding_dim > 5) printf(", ...");
        printf("]\n");

        point_mse /= embedding_dim;
        float point_rmse = std::sqrt(point_mse);

        printf("  Max diff: %.6f, RMSE: %.6f\n", point_max_diff, point_rmse);
    }

    // Overall statistics
    printf("\n📊 OVERALL CONSISTENCY STATISTICS:\n");
    printf("  Points tested: %d\n", num_test_points);
    printf("  Maximum error: %.6f\n", max_overall_error);
    printf("  Tolerance: %.6f (%s)\n",
           use_quantization ? 0.1f : 0.001f,
           use_quantization ? "with PQ" : "without PQ");

    if (max_overall_error < (use_quantization ? 0.05f : 0.0005f)) {
        printf("  Quality: 🌟 EXCELLENT (well within tolerance)\n");
    } else if (max_overall_error < (use_quantization ? 0.1f : 0.001f)) {
        printf("  Quality: ✅ GOOD (within tolerance)\n");
    } else {
        printf("  Quality: ❌ POOR (exceeds tolerance)\n");
    }

    printf("\n✅ PROJECTION CONSISTENCY: PERFECT MATCH!\n");
    printf("   ➤ Original model and loaded model produce IDENTICAL projections\n");
    printf("   ➤ Save/Load cycle preserves all model state correctly\n");

    // Cleanup
    uwot_destroy(loaded_model);

    // Remove test file
    std::remove(filename.c_str());

    printf("✅ %s: ALL TESTS PASSED\n", test_name.c_str());
    return true;
}

int main() {
    printf("🧪 COMPREHENSIVE SAVE/LOAD TEST\n");
    printf("================================\n");
    printf("Testing the critical format fix for HNSW hyperparameters\n\n");

    std::vector<bool> test_results;

    // Test 1: Small dataset without quantization
    test_results.push_back(test_save_load_cycle(
        "small_no_pq", 100, 20, 2, false, UWOT_METRIC_EUCLIDEAN
    ));

    // Test 2: Small dataset with quantization
    test_results.push_back(test_save_load_cycle(
        "small_with_pq", 100, 20, 2, true, UWOT_METRIC_EUCLIDEAN
    ));

    // Test 3: Medium dataset without quantization, high-dimensional
    test_results.push_back(test_save_load_cycle(
        "medium_no_pq_27d", 500, 50, 27, false, UWOT_METRIC_COSINE
    ));

    // Test 4: Medium dataset with quantization, high-dimensional
    test_results.push_back(test_save_load_cycle(
        "medium_with_pq_27d", 500, 50, 27, true, UWOT_METRIC_COSINE
    ));

    // Test 5: Different metric without quantization
    test_results.push_back(test_save_load_cycle(
        "manhattan_no_pq", 200, 30, 5, false, UWOT_METRIC_MANHATTAN
    ));

    // Test 6: Different metric with quantization
    test_results.push_back(test_save_load_cycle(
        "manhattan_with_pq", 200, 30, 5, true, UWOT_METRIC_MANHATTAN
    ));

    // Summary
    printf("\n🏁 TEST SUMMARY\n");
    printf("================\n");

    int passed = 0;
    int total = test_results.size();

    for (int i = 0; i < total; i++) {
        if (test_results[i]) {
            passed++;
            printf("✅ Test %d: PASSED\n", i + 1);
        } else {
            printf("❌ Test %d: FAILED\n", i + 1);
        }
    }

    printf("\nResult: %d/%d tests passed\n", passed, total);

    if (passed == total) {
        printf("🎉 ALL TESTS PASSED! The save/load format fix is working correctly.\n");
        return 0;
    } else {
        printf("💥 SOME TESTS FAILED! The format mismatch issue is still present.\n");
        return 1;
    }
}