// Test HNSW save/load consistency fixes
// This test validates that the implemented fixes resolve the discrepancy
// between original fit and loaded model transform results

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "uwot_pure_cpp/uwot_simple_wrapper.h"

// Generate simple test data
std::vector<std::vector<float>> generate_test_data(int n_samples, int n_features, int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> data(n_samples, std::vector<float>(n_features));
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            data[i][j] = dist(gen);
        }
    }
    return data;
}

// Convert to flat array
std::vector<float> flatten_data(const std::vector<std::vector<float>>& data) {
    std::vector<float> flat;
    for (const auto& row : data) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

// Calculate Mean Squared Error between two embeddings
double calculate_mse(const std::vector<float>& embed1, const std::vector<float>& embed2) {
    if (embed1.size() != embed2.size()) {
        return -1.0; // Error
    }

    double sum_sq_diff = 0.0;
    for (size_t i = 0; i < embed1.size(); i++) {
        double diff = embed1[i] - embed2[i];
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / embed1.size();
}

int main() {
    printf("🧪 Testing HNSW Save/Load Consistency Fixes\n");
    printf("==========================================\n\n");

    try {
        // Generate test data
        auto data = generate_test_data(1000, 50, 42);
        auto flat_data = flatten_data(data);

        printf("📊 Test Setup:\n");
        printf("  - Dataset: %zu samples × %zu features\n", data.size(), data[0].size());
        printf("  - Test point: First sample from training data\n\n");

        // Step 1: Train UMAP model
        printf("🔧 Step 1: Training UMAP model...\n");
        UwotModel* model = uwot_create();
        if (!model) {
            printf("❌ Failed to create model!\n");
            return 1;
        }

        std::vector<float> embedding(data.size() * 2);
        int train_result = uwot_fit_with_progress(
            model,
            flat_data.data(),
            static_cast<int>(data.size()),
            static_cast<int>(data[0].size()),
            2,        // 2D embedding
            15,       // n_neighbors
            0.1f,     // min_dist
            1.0f,     // spread
            100,      // epochs (reduced for faster testing)
            UWOT_METRIC_EUCLIDEAN,
            embedding.data(),
            nullptr,  // no callback
            0,        // force exact knn = false (use HNSW)
            16,       // hnsw_M
            200,      // hnsw_ef_construction
            64        // hnsw_ef_search
        );

        if (train_result != UWOT_SUCCESS) {
            printf("❌ Failed to train UMAP model: %d\n", train_result);
            uwot_destroy(model);
            return 1;
        }

        printf("✅ Training completed successfully\n\n");

        // Step 2: Transform test point with original model
        printf("🔧 Step 2: Transform with original model...\n");
        std::vector<float> test_point(data[0]);
        std::vector<float> original_result(2);

        int transform_result = uwot_transform(
            model,
            test_point.data(),
            1,  // n_samples
            static_cast<int>(data[0].size()),
            original_result.data()
        );

        if (transform_result != UWOT_SUCCESS) {
            printf("❌ Original transform failed: %d\n", transform_result);
            uwot_destroy(model);
            return 1;
        }
        printf("✅ Original transform: [%.6f, %.6f]\n", original_result[0], original_result[1]);

        // Step 3: Save model
        printf("\n🔧 Step 3: Saving model...\n");
        const char* model_path = "test_save_load_fix.umap";
        int save_result = uwot_save_model(model, model_path);
        if (save_result != UWOT_SUCCESS) {
            printf("❌ Failed to save model: %d\n", save_result);
            uwot_destroy(model);
            return 1;
        }
        printf("✅ Model saved successfully\n");

        // Free original model
        uwot_destroy(model);

        // Step 4: Load model
        printf("\n🔧 Step 4: Loading model...\n");
        UwotModel* loaded_model = uwot_load_model(model_path);
        if (!loaded_model) {
            printf("❌ Failed to load model!\n");
            return 1;
        }
        printf("✅ Model loaded successfully\n");

        // Step 5: Transform test point with loaded model
        printf("\n🔧 Step 5: Transform with loaded model...\n");
        std::vector<float> loaded_result(2);

        transform_result = uwot_transform(
            loaded_model,
            test_point.data(),
            1,  // n_samples
            static_cast<int>(data[0].size()),
            loaded_result.data()
        );

        if (transform_result != UWOT_SUCCESS) {
            printf("❌ Loaded model transform failed: %d\n", transform_result);
            uwot_destroy(loaded_model);
            return 1;
        }
        printf("✅ Loaded transform: [%.6f, %.6f]\n", loaded_result[0], loaded_result[1]);

        // Step 6: Compare results
        printf("\n🔍 Step 6: Analyzing consistency...\n");
        double mse = calculate_mse(original_result, loaded_result);

        float max_diff = 0.0f;
        for (int i = 0; i < 2; i++) {
            float diff = std::abs(original_result[i] - loaded_result[i]);
            max_diff = std::max(max_diff, diff);
        }

        printf("📊 Consistency Analysis:\n");
        printf("  - Original result: [%.6f, %.6f]\n", original_result[0], original_result[1]);
        printf("  - Loaded result:   [%.6f, %.6f]\n", loaded_result[0], loaded_result[1]);
        printf("  - Maximum difference: %.6f\n", max_diff);
        printf("  - Mean Squared Error: %.8f\n", mse);

        // Determine success
        const float tolerance = 0.001f;  // Very strict tolerance for exact reproducibility
        bool success = max_diff < tolerance;

        printf("\n🎯 FINAL RESULT:\n");
        if (success) {
            printf("✅ SUCCESS: Transform consistency achieved!\n");
            printf("   The HNSW save/load fixes are working correctly.\n");
            printf("   Maximum difference (%.6f) is below tolerance (%.6f)\n", max_diff, tolerance);
        } else {
            printf("❌ FAILURE: Transform inconsistency detected!\n");
            printf("   Maximum difference (%.6f) exceeds tolerance (%.6f)\n", max_diff, tolerance);
            printf("   The fixes may need further refinement.\n");
        }

        // Cleanup
        uwot_destroy(loaded_model);

        printf("\n🧹 Cleanup completed\n");
        return success ? 0 : 1;

    } catch (const std::exception& e) {
        printf("❌ Exception occurred: %s\n", e.what());
        return 1;
    } catch (...) {
        printf("❌ Unknown exception occurred\n");
        return 1;
    }
}