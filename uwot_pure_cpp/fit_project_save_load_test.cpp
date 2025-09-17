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

bool test_cycle(const std::string& name, bool use_quantization, bool force_exact) {
    printf("\n=== %s ===\n", name.c_str());
    printf("PQ: %s, k-NN: %s\n",
           use_quantization ? "ON" : "OFF",
           force_exact ? "EXACT" : "HNSW");

    // Small dataset for testing
    int n_samples = 100;
    int n_features = 20;
    int embedding_dim = 2;
    int n_neighbors = 10;
    int n_epochs = 50;

    auto data = generate_data(n_samples, n_features);

    // 1. FIT: Train model with PROPER parameter control
    UwotModel* model = uwot_create();
    if (!model) {
        printf("❌ Failed to create model\n");
        return false;
    }

    printf("1. Training model with PQ=%s, k-NN=%s...\n",
           use_quantization ? "ON" : "OFF",
           force_exact ? "EXACT" : "HNSW");

    std::vector<float> embedding(n_samples * embedding_dim);

    // Use uwot_fit_with_enhanced_progress to control quantization and k-NN method
    int result = uwot_fit_with_enhanced_progress(
        model, data.data(), n_samples, n_features,
        embedding_dim, n_neighbors, 0.1f, 1.0f, n_epochs,
        UWOT_METRIC_EUCLIDEAN,
        embedding.data(),
        nullptr, // progress callback
        force_exact ? 1 : 0,     // force_exact_knn
        use_quantization ? 1 : 0, // use_quantization
        16,   // M (HNSW parameter)
        64,   // ef_construction
        32,   // ef_search
        4     // pq_subspaces
    );

    if (result != UWOT_SUCCESS) {
        printf("❌ Training failed: %d\n", result);
        uwot_destroy(model);
        return false;
    }
    printf("   ✅ Training completed\n");

    // Display original model info
    int n_vertices, n_dim, emb_dim, n_neigh, use_pq, hnsw_M, hnsw_ef_c, hnsw_ef_s;
    float min_d, spr;
    UwotMetric met;
    uwot_get_model_info(model, &n_vertices, &n_dim, &emb_dim, &n_neigh,
                       &min_d, &spr, &met, &use_pq, &hnsw_M, &hnsw_ef_c, &hnsw_ef_s);
    printf("   Original model: %dD→%dD, PQ=%s, HNSW(M=%d)\n",
           n_dim, emb_dim, use_pq ? "ON" : "OFF", hnsw_M);

    // 2. PROJECT: Test projection with original model
    std::vector<float> test_point(n_features);
    for (int i = 0; i < n_features; i++) {
        test_point[i] = data[i]; // First sample
    }

    std::vector<float> orig_projection(embedding_dim);
    result = uwot_transform(model, test_point.data(), 1, n_features, orig_projection.data());

    if (result != UWOT_SUCCESS) {
        printf("❌ Original projection failed: %d\n", result);
        uwot_destroy(model);
        return false;
    }

    printf("2. Original projection: [%.6f, %.6f]\n", orig_projection[0], orig_projection[1]);

    // 3. SAVE: Save the model
    std::string filename = "test_" + name + ".umap";
    printf("3. Saving to %s...\n", filename.c_str());

    result = uwot_save_model(model, filename.c_str());
    if (result != UWOT_SUCCESS) {
        printf("❌ Save failed: %d\n", result);
        uwot_destroy(model);
        return false;
    }
    printf("   ✅ Model saved\n");

    // Destroy original model
    uwot_destroy(model);

    // 4. LOAD: Load the model
    printf("4. Loading from %s...\n", filename.c_str());
    UwotModel* loaded_model = uwot_load_model(filename.c_str());
    if (!loaded_model) {
        printf("❌ Load failed\n");
        return false;
    }
    printf("   ✅ Model loaded\n");

    // Display loaded model info to verify parameters match
    int load_n_vertices, load_n_dim, load_emb_dim, load_n_neigh, load_use_pq;
    int load_hnsw_M, load_hnsw_ef_c, load_hnsw_ef_s;
    float load_min_d, load_spr;
    UwotMetric load_met;
    uwot_get_model_info(loaded_model, &load_n_vertices, &load_n_dim, &load_emb_dim, &load_n_neigh,
                       &load_min_d, &load_spr, &load_met, &load_use_pq, &load_hnsw_M, &load_hnsw_ef_c, &load_hnsw_ef_s);
    printf("   Loaded model:  %dD→%dD, PQ=%s, HNSW(M=%d)\n",
           load_n_dim, load_emb_dim, load_use_pq ? "ON" : "OFF", load_hnsw_M);

    // Verify critical parameters match
    bool params_match = (use_pq == load_use_pq) && (hnsw_M == load_hnsw_M) &&
                       (emb_dim == load_emb_dim) && (n_dim == load_n_dim);
    if (!params_match) {
        printf("❌ Model parameters don't match after load!\n");
        uwot_destroy(loaded_model);
        return false;
    }
    printf("   ✅ Parameters match\n");

    // 5. PROJECT: Test projection with loaded model
    std::vector<float> loaded_projection(embedding_dim);
    result = uwot_transform(loaded_model, test_point.data(), 1, n_features, loaded_projection.data());

    if (result != UWOT_SUCCESS) {
        printf("❌ Loaded projection failed: %d\n", result);
        uwot_destroy(loaded_model);
        return false;
    }

    printf("5. Loaded projection:  [%.6f, %.6f]\n", loaded_projection[0], loaded_projection[1]);

    // 6. COMPARE: Check if projections match
    float diff_x = std::abs(orig_projection[0] - loaded_projection[0]);
    float diff_y = std::abs(orig_projection[1] - loaded_projection[1]);
    float max_diff = std::max(diff_x, diff_y);

    printf("6. Difference:         [%.6f, %.6f] max=%.6f\n", diff_x, diff_y, max_diff);

    const float tolerance = use_quantization ? 0.1f : 0.001f;
    bool success = max_diff < tolerance;

    if (success) {
        printf("✅ SUCCESS: Projections match (%.6f < %.6f)\n", max_diff, tolerance);
    } else {
        printf("❌ FAILED: Projections differ (%.6f >= %.6f)\n", max_diff, tolerance);
    }

    // Cleanup
    uwot_destroy(loaded_model);
    std::remove(filename.c_str());

    return success;
}

int main() {
    printf("🔄 FIT → PROJECT → SAVE → LOAD → PROJECT → COMPARE TEST\n");
    printf("========================================================\n");

    std::vector<bool> results;

    // Test 4 scenarios
    results.push_back(test_cycle("no_pq_hnsw", false, false));   // No PQ, HNSW
    results.push_back(test_cycle("no_pq_exact", false, true));   // No PQ, Exact
    results.push_back(test_cycle("with_pq_hnsw", true, false));  // PQ, HNSW
    results.push_back(test_cycle("with_pq_exact", true, true));  // PQ, Exact

    printf("\n🏁 FINAL RESULTS\n");
    printf("================\n");

    const char* names[] = {
        "No PQ + HNSW",
        "No PQ + Exact",
        "PQ + HNSW",
        "PQ + Exact"
    };

    int passed = 0;
    for (int i = 0; i < 4; i++) {
        if (results[i]) {
            printf("✅ %s: PASSED\n", names[i]);
            passed++;
        } else {
            printf("❌ %s: FAILED\n", names[i]);
        }
    }

    printf("\nResult: %d/4 tests passed\n", passed);

    if (passed == 4) {
        printf("🎉 ALL TESTS PASSED! Save/load cycle works perfectly!\n");
        return 0;
    } else {
        printf("💥 %d tests failed. Fix needed.\n", 4 - passed);
        return 1;
    }
}