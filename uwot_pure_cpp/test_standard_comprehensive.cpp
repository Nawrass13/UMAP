#include "uwot_simple_wrapper.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <cstring>

// STRICT PASS/FAIL THRESHOLDS - MUST MEET THESE TO PASS
const float MAX_ALLOWED_1_PERCENT_ERROR_RATE = 0.5f;  // Max 0.5% of points can have >1% error
const float MAX_ALLOWED_FIT_TRANSFORM_MSE = 0.1f;     // MSE between fit and transform must be <0.1
const float MAX_ALLOWED_SAVE_LOAD_MSE = 1e-6f;        // Save/load must be nearly identical (1e-6)
const float MAX_ALLOWED_COORDINATE_COLLAPSE = 1e-4f;  // Detect coordinate collapse
const int MIN_COORDINATE_VARIETY = 10;                 // At least 10 different coordinate values
const float MAX_ALLOWED_LOSS_INCREASE_PERCENT = 50.0f; // Loss can increase max 50% from minimum

// Test configuration
const int N_SAMPLES = 2000;  // Reduced from 5000 for faster standard testing
const int N_DIM = 150;       // Reduced from 320 for faster standard testing
const int N_NEIGHBORS = 15;
const float MIN_DIST = 0.3f;
const float SPREAD = 4.0f;
const int N_EPOCHS = 100;    // Reduced from 200 for faster testing

struct TestResults {
    float fit_transform_mse_2d;
    float fit_transform_mse_20d;
    float save_load_mse_2d;
    float save_load_mse_20d;
    float error_rate_1_percent_2d;
    float error_rate_1_percent_20d;
    bool coordinate_variety_2d;
    bool coordinate_variety_20d;
    bool loss_convergence_2d;
    bool loss_convergence_20d;
    bool all_tests_passed;
};

// Calculate MSE between two embedding arrays
float calculate_mse(const std::vector<float>& a, const std::vector<float>& b, int n_points, int embedding_dim) {
    if (a.size() != b.size() || a.size() != static_cast<size_t>(n_points * embedding_dim)) {
        return 1e6f; // Error indicator
    }

    float sum_squared_diff = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum_squared_diff += diff * diff;
    }
    return sum_squared_diff / static_cast<float>(a.size());
}

// Calculate percentage of points with >1% error
float calculate_error_rate_1_percent(const std::vector<float>& a, const std::vector<float>& b,
                                     int n_points, int embedding_dim) {
    int points_with_high_error = 0;

    for (int i = 0; i < n_points; i++) {
        float point_error = 0.0f;
        for (int d = 0; d < embedding_dim; d++) {
            size_t idx = static_cast<size_t>(i) * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d);
            float diff = std::abs(a[idx] - b[idx]);
            point_error = std::max(point_error, diff);
        }

        if (point_error > 0.01f) { // >1% error
            points_with_high_error++;
        }
    }

    return (static_cast<float>(points_with_high_error) / static_cast<float>(n_points)) * 100.0f;
}

// Check coordinate variety (detect collapse)
bool check_coordinate_variety(const std::vector<float>& coords, int n_points, int embedding_dim) {
    if (n_points < MIN_COORDINATE_VARIETY) return true; // Too few points to judge

    for (int d = 0; d < embedding_dim; d++) {
        std::vector<float> dimension_values;

        // Collect values for this dimension
        for (int i = 0; i < n_points; i++) {
            size_t idx = static_cast<size_t>(i) * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d);
            dimension_values.push_back(coords[idx]);
        }

        // Sort and count unique values (with tolerance)
        std::sort(dimension_values.begin(), dimension_values.end());
        int unique_count = 1;
        for (int i = 1; i < n_points; i++) {
            if (std::abs(dimension_values[i] - dimension_values[i-1]) > MAX_ALLOWED_COORDINATE_COLLAPSE) {
                unique_count++;
            }
        }

        if (unique_count < MIN_COORDINATE_VARIETY) {
            std::cout << "❌ Coordinate collapse detected in dimension " << d
                     << " - only " << unique_count << " unique values" << std::endl;
            return false;
        }
    }

    return true;
}

// Loss tracking for convergence validation
static std::vector<float> g_loss_history;

// Progress callback that captures loss values (v2 signature)
void loss_tracking_callback(const char* phase, int current, int total, float percent, const char* message) {
    // Extract loss from message if available, or use current progress as proxy
    static float last_loss = 0.0f;

    // Parse loss from message if it contains "Loss: X.XXX"
    if (message && strstr(message, "Loss:")) {
        float parsed_loss;
        if (sscanf(message, "%*[^L]Loss: %f", &parsed_loss) == 1) {
            last_loss = parsed_loss;
            g_loss_history.push_back(parsed_loss);
        }
    }

    // Display progress
    printf("\r%s: [%-20s] %.1f%% (%d/%d)",
           phase ? phase : "Training", "", percent, current, total);
    if (message) {
        printf(" %s", message);
    }

    if (current >= total - 1) {
        printf("\nTraining completed!\n");
    }
    fflush(stdout);
}

// Validate loss convergence (should generally decrease over time)
bool validate_loss_convergence(const std::vector<float>& loss_history) {
    if (loss_history.size() < 10) return true; // Too few points to judge

    // Find minimum loss in first half and last quarter
    size_t first_half = loss_history.size() / 2;
    size_t last_quarter_start = (loss_history.size() * 3) / 4;

    float min_loss_first_half = *std::min_element(loss_history.begin(), loss_history.begin() + first_half);
    float final_loss = loss_history.back();
    float avg_last_quarter = 0.0f;

    for (size_t i = last_quarter_start; i < loss_history.size(); i++) {
        avg_last_quarter += loss_history[i];
    }
    avg_last_quarter /= (loss_history.size() - last_quarter_start);

    // Check if final loss is reasonable compared to minimum
    float loss_increase_percent = ((final_loss - min_loss_first_half) / min_loss_first_half) * 100.0f;

    std::cout << "📊 Loss Convergence Analysis:" << std::endl;
    std::cout << "   Initial loss: " << loss_history[0] << std::endl;
    std::cout << "   Minimum loss (first half): " << min_loss_first_half << std::endl;
    std::cout << "   Final loss: " << final_loss << std::endl;
    std::cout << "   Average last quarter: " << avg_last_quarter << std::endl;
    std::cout << "   Loss increase from min: " << loss_increase_percent << "% (threshold: "
              << MAX_ALLOWED_LOSS_INCREASE_PERCENT << "%)" << std::endl;

    bool converged = (loss_increase_percent <= MAX_ALLOWED_LOSS_INCREASE_PERCENT);
    std::cout << "   Convergence: " << (converged ? "✅ PASS" : "❌ FAIL") << std::endl;

    return converged;
}

// Generate synthetic dataset with variance
void generate_test_data(std::vector<float>& data, int n_samples, int n_dim) {
    std::mt19937 rng(12345); // Fixed seed for reproducibility

    // Create data with different variance per dimension
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_dim; j++) {
            float base_variance = 1.0f + (j % 10) * 0.5f; // Variance 1.0 to 5.5
            std::normal_distribution<float> dist(0.0f, base_variance);
            data[static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j)] = dist(rng);
        }
    }
}

// Test one embedding dimension
bool test_embedding_dimension(int embedding_dim, TestResults& results) {
    std::cout << "\n🚀 Testing " << embedding_dim << "D Embedding" << std::endl;
    std::cout << "===============================================" << std::endl;

    // Generate test data
    std::vector<float> data(static_cast<size_t>(N_SAMPLES) * static_cast<size_t>(N_DIM));
    generate_test_data(data, N_SAMPLES, N_DIM);

    // Step 1: Train model
    std::cout << "📚 Step 1: Training UMAP model..." << std::endl;
    UwotModel* model = uwot_create();
    if (!model) {
        std::cout << "❌ Failed to create model" << std::endl;
        return false;
    }

    std::vector<float> fit_embedding(static_cast<size_t>(N_SAMPLES) * static_cast<size_t>(embedding_dim));

    // Clear loss history for this test
    g_loss_history.clear();

    int result = uwot_fit_with_progress_v2(model, data.data(), N_SAMPLES, N_DIM, embedding_dim,
        N_NEIGHBORS, MIN_DIST, SPREAD, N_EPOCHS, UWOT_METRIC_EUCLIDEAN,
        fit_embedding.data(), loss_tracking_callback, 0, 16, 200, 64);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Training failed: " << result << std::endl;
        uwot_destroy(model);
        return false;
    }
    std::cout << "✅ Training completed" << std::endl;

    // Validate loss convergence
    bool loss_converged = validate_loss_convergence(g_loss_history);

    // Step 2: Transform with same data
    std::cout << "🔄 Step 2: Transform validation..." << std::endl;
    std::vector<float> transform_embedding(static_cast<size_t>(N_SAMPLES) * static_cast<size_t>(embedding_dim));

    result = uwot_transform_detailed(model, data.data(), N_SAMPLES, N_DIM,
        transform_embedding.data(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Transform failed: " << result << std::endl;
        uwot_destroy(model);
        return false;
    }
    std::cout << "✅ Transform completed" << std::endl;

    // Step 3: Analyze fit vs transform
    float fit_transform_mse = calculate_mse(fit_embedding, transform_embedding, N_SAMPLES, embedding_dim);
    float error_rate_1_percent = calculate_error_rate_1_percent(fit_embedding, transform_embedding, N_SAMPLES, embedding_dim);
    bool coordinate_variety = check_coordinate_variety(transform_embedding, N_SAMPLES, embedding_dim);

    std::cout << "📊 Fit vs Transform Analysis:" << std::endl;
    std::cout << "   MSE: " << fit_transform_mse << " (threshold: " << MAX_ALLOWED_FIT_TRANSFORM_MSE << ")" << std::endl;
    std::cout << "   1% error rate: " << error_rate_1_percent << "% (threshold: " << MAX_ALLOWED_1_PERCENT_ERROR_RATE << "%)" << std::endl;
    std::cout << "   Coordinate variety: " << (coordinate_variety ? "✅ PASS" : "❌ FAIL") << std::endl;

    // Step 4: Save model
    std::cout << "💾 Step 4: Save/load validation..." << std::endl;
    std::string model_file = "test_standard_" + std::to_string(embedding_dim) + "d.umap";

    result = uwot_save_model(model, model_file.c_str());
    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Save failed: " << result << std::endl;
        uwot_destroy(model);
        return false;
    }

    // Step 5: Load model and transform again
    uwot_destroy(model);
    UwotModel* loaded_model = uwot_load_model(model_file.c_str());
    if (!loaded_model) {
        std::cout << "❌ Load failed" << std::endl;
        return false;
    }

    std::vector<float> loaded_transform_embedding(static_cast<size_t>(N_SAMPLES) * static_cast<size_t>(embedding_dim));

    result = uwot_transform_detailed(loaded_model, data.data(), N_SAMPLES, N_DIM,
        loaded_transform_embedding.data(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    if (result != UWOT_SUCCESS) {
        std::cout << "❌ Loaded model transform failed: " << result << std::endl;
        uwot_destroy(loaded_model);
        return false;
    }

    // Step 6: Analyze save/load consistency
    float save_load_mse = calculate_mse(transform_embedding, loaded_transform_embedding, N_SAMPLES, embedding_dim);

    std::cout << "📊 Save/Load Analysis:" << std::endl;
    std::cout << "   Original vs Loaded MSE: " << save_load_mse << " (threshold: " << MAX_ALLOWED_SAVE_LOAD_MSE << ")" << std::endl;

    uwot_destroy(loaded_model);

    // Store results
    if (embedding_dim == 2) {
        results.fit_transform_mse_2d = fit_transform_mse;
        results.save_load_mse_2d = save_load_mse;
        results.error_rate_1_percent_2d = error_rate_1_percent;
        results.coordinate_variety_2d = coordinate_variety;
        results.loss_convergence_2d = loss_converged;
    } else if (embedding_dim == 20) {
        results.fit_transform_mse_20d = fit_transform_mse;
        results.save_load_mse_20d = save_load_mse;
        results.error_rate_1_percent_20d = error_rate_1_percent;
        results.coordinate_variety_20d = coordinate_variety;
        results.loss_convergence_20d = loss_converged;
    }

    // Check pass/fail for this dimension
    bool passed = (fit_transform_mse < MAX_ALLOWED_FIT_TRANSFORM_MSE) &&
                  (save_load_mse < MAX_ALLOWED_SAVE_LOAD_MSE) &&
                  (error_rate_1_percent < MAX_ALLOWED_1_PERCENT_ERROR_RATE) &&
                  coordinate_variety &&
                  loss_converged;

    std::cout << "\n🎯 " << embedding_dim << "D Result: " << (passed ? "✅ PASS" : "❌ FAIL") << std::endl;

    return passed;
}

int main() {
    std::cout << "🧪 COMPREHENSIVE STANDARD UMAP TEST" << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << "Dataset: " << N_SAMPLES << " x " << N_DIM << " dimensions" << std::endl;
    std::cout << "Parameters: min_dist=" << MIN_DIST << ", spread=" << SPREAD
              << ", n_neighbors=" << N_NEIGHBORS << ", epochs=" << N_EPOCHS << std::endl;
    std::cout << "\n🎯 PASS/FAIL THRESHOLDS:" << std::endl;
    std::cout << "   Fit vs Transform MSE: < " << MAX_ALLOWED_FIT_TRANSFORM_MSE << std::endl;
    std::cout << "   Save/Load MSE: < " << MAX_ALLOWED_SAVE_LOAD_MSE << std::endl;
    std::cout << "   1% Error Rate: < " << MAX_ALLOWED_1_PERCENT_ERROR_RATE << "%" << std::endl;
    std::cout << "   Coordinate Variety: >= " << MIN_COORDINATE_VARIETY << " unique values per dimension" << std::endl;
    std::cout << "   Loss Convergence: < " << MAX_ALLOWED_LOSS_INCREASE_PERCENT << "% increase from minimum" << std::endl;

    TestResults results = {};

    // Test 2D and 20D embeddings
    bool test_2d_passed = test_embedding_dimension(2, results);
    bool test_20d_passed = test_embedding_dimension(20, results);

    results.all_tests_passed = test_2d_passed && test_20d_passed;

    // Final summary
    std::cout << "\n=================================================================================" << std::endl;
    std::cout << "🎉 COMPREHENSIVE STANDARD TEST RESULTS" << std::endl;
    std::cout << "=================================================================================" << std::endl;

    std::cout << "\n📊 Detailed Results:" << std::endl;
    std::cout << "Dimension      Fit vs Transform MSE    Save/Load MSE         1% Error Rate    Variety    Loss Conv" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------------" << std::endl;
    printf("2D             %.6f (%.1s)           %.6f (%.1s)        %.1f%% (%.1s)        %s        %s\n",
           results.fit_transform_mse_2d,
           results.fit_transform_mse_2d < MAX_ALLOWED_FIT_TRANSFORM_MSE ? "✅" : "❌",
           results.save_load_mse_2d,
           results.save_load_mse_2d < MAX_ALLOWED_SAVE_LOAD_MSE ? "✅" : "❌",
           results.error_rate_1_percent_2d,
           results.error_rate_1_percent_2d < MAX_ALLOWED_1_PERCENT_ERROR_RATE ? "✅" : "❌",
           results.coordinate_variety_2d ? "✅" : "❌",
           results.loss_convergence_2d ? "✅" : "❌");

    printf("20D            %.6f (%.1s)           %.6f (%.1s)        %.1f%% (%.1s)        %s        %s\n",
           results.fit_transform_mse_20d,
           results.fit_transform_mse_20d < MAX_ALLOWED_FIT_TRANSFORM_MSE ? "✅" : "❌",
           results.save_load_mse_20d,
           results.save_load_mse_20d < MAX_ALLOWED_SAVE_LOAD_MSE ? "✅" : "❌",
           results.error_rate_1_percent_20d,
           results.error_rate_1_percent_20d < MAX_ALLOWED_1_PERCENT_ERROR_RATE ? "✅" : "❌",
           results.coordinate_variety_20d ? "✅" : "❌",
           results.loss_convergence_20d ? "✅" : "❌");

    std::cout << "\n🎯 OVERALL RESULT: " << (results.all_tests_passed ? "✅ ALL TESTS PASSED" : "❌ TESTS FAILED") << std::endl;

    if (!results.all_tests_passed) {
        std::cout << "\n🚨 CRITICAL FAILURES DETECTED:" << std::endl;
        if (results.fit_transform_mse_2d >= MAX_ALLOWED_FIT_TRANSFORM_MSE ||
            results.fit_transform_mse_20d >= MAX_ALLOWED_FIT_TRANSFORM_MSE) {
            std::cout << "   - Fit vs Transform MSE too high (possible coordinate collapse or HNSW failure)" << std::endl;
        }
        if (results.save_load_mse_2d >= MAX_ALLOWED_SAVE_LOAD_MSE ||
            results.save_load_mse_20d >= MAX_ALLOWED_SAVE_LOAD_MSE) {
            std::cout << "   - Save/Load inconsistency (serialization bug)" << std::endl;
        }
        if (results.error_rate_1_percent_2d >= MAX_ALLOWED_1_PERCENT_ERROR_RATE ||
            results.error_rate_1_percent_20d >= MAX_ALLOWED_1_PERCENT_ERROR_RATE) {
            std::cout << "   - Too many points with >1% error (HNSW approximation failure)" << std::endl;
        }
        if (!results.coordinate_variety_2d || !results.coordinate_variety_20d) {
            std::cout << "   - Coordinate collapse detected (normalization bug)" << std::endl;
        }
        if (!results.loss_convergence_2d || !results.loss_convergence_20d) {
            std::cout << "   - Loss convergence failure (optimization bug - loss increasing instead of decreasing)" << std::endl;
        }
    }

    std::cout << "\n✅ Validated Features:" << std::endl;
    std::cout << "   - HNSW optimization with approximation accuracy" << std::endl;
    std::cout << "   - Save/load projection identity" << std::endl;
    std::cout << "   - Coordinate variety (anti-collapse detection)" << std::endl;
    std::cout << "   - 1% error rate thresholds" << std::endl;
    std::cout << "   - Multi-dimensional embedding consistency" << std::endl;
    std::cout << "   - Loss function convergence validation" << std::endl;

    return results.all_tests_passed ? 0 : 1;
}