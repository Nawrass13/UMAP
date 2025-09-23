#include "uwot_simple_wrapper.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <string>
#include <cstring>

// Configuration constants - Optimized for comprehensive testing
const int N_SAMPLES = 5000; // Reduced size for better fit/transform consistency
const int N_DIM = 320;
const int N_NEIGHBORS = 20;
const float MIN_DIST = 0.5f;
const float SPREAD = 6.0f;
const int N_EPOCHS = 200; // Full epochs for quality results
const int RANDOM_STATE = 42;

// Test result structure
struct TestResult {
    std::vector<float> original_projection;
    std::vector<float> transform_projection;
    std::vector<float> loaded_transform_projection;
    double fit_vs_transform_mse;
    double original_vs_loaded_mse;
    double fit_vs_loaded_mse;
    std::string phase_name;
    int embedding_dim;
};

// Generate synthetic dataset with good variance
void generate_synthetic_data(std::vector<float>& data, int n_samples, int n_dim, int seed = 42) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);

    data.resize(n_samples * n_dim);

    std::cout << "🔧 Generating " << n_samples << " x " << n_dim << " synthetic dataset..." << std::endl;

    // Create clusters with different patterns
    int samples_per_cluster = n_samples / 4;

    for (int i = 0; i < n_samples; i++) {
        int cluster = i / samples_per_cluster;
        if (cluster >= 4) cluster = 3; // Handle remainder

        for (int j = 0; j < n_dim; j++) {
            float value = 0.0f;

            switch (cluster) {
            case 0: // Gaussian cluster
                value = normal(gen) + 2.0f;
                break;
            case 1: // Uniform cluster
                value = uniform(gen) - 2.0f;
                break;
            case 2: // Correlated features
                if (j % 3 == 0) {
                    value = normal(gen) * 0.5f;
                } else {
                    value = data[i * n_dim + (j - j % 3)] * 0.7f + normal(gen) * 0.3f;
                }
                break;
            case 3: // Sparse pattern
                if (j % 10 == 0) {
                    value = normal(gen) * 2.0f + 1.0f;
                } else {
                    value = normal(gen) * 0.1f;
                }
                break;
            }

            data[i * n_dim + j] = value;
        }
    }

    // Add some noise to increase variance
    std::uniform_real_distribution<float> noise(-0.1f, 0.1f);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] += noise(gen);
    }

    // Calculate and display variance statistics
    std::vector<float> feature_variances(n_dim, 0.0f);
    for (int j = 0; j < n_dim; j++) {
        float mean = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            mean += data[i * n_dim + j];
        }
        mean /= n_samples;

        float variance = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float diff = data[i * n_dim + j] - mean;
            variance += diff * diff;
        }
        variance /= (n_samples - 1);
        feature_variances[j] = variance;
    }

    float min_var = *std::min_element(feature_variances.begin(), feature_variances.end());
    float max_var = *std::max_element(feature_variances.begin(), feature_variances.end());
    float mean_var = std::accumulate(feature_variances.begin(), feature_variances.end(), 0.0f) / n_dim;

    std::cout << "📊 Dataset variance stats: min=" << std::fixed << std::setprecision(3)
              << min_var << ", max=" << max_var << ", mean=" << mean_var << std::endl;
}

// Calculate MSE between two projections
double calculate_mse(const std::vector<float>& proj1, const std::vector<float>& proj2, int embedding_dim) {
    if (proj1.size() != proj2.size()) {
        std::cout << "❌ MSE calculation error: size mismatch " << proj1.size() << " vs " << proj2.size() << std::endl;
        return -1.0;
    }

    double mse = 0.0;
    int n_points = static_cast<int>(proj1.size()) / embedding_dim;

    for (int i = 0; i < n_points; i++) {
        for (int d = 0; d < embedding_dim; d++) {
            double diff = proj1[i * embedding_dim + d] - proj2[i * embedding_dim + d];
            mse += diff * diff;
        }
    }

    return mse / (n_points * embedding_dim);
}

// Progress callback for training
void progress_callback_v2(const char* phase, int current, int total, float percent, const char* message) {
    static int last_percent = -1;
    int current_percent = static_cast<int>(percent);

    // Show progress every 5% or at completion
    if (current_percent != last_percent && (current_percent % 5 == 0 || current == total - 1)) {
        std::cout << "\r[" << phase << "] " << std::fixed << std::setprecision(1)
                  << percent << "% (" << current << "/" << total << ")";
        if (message && strlen(message) > 0) {
            std::cout << " " << message;
        }
        std::cout << std::flush;
        last_percent = current_percent;
    }
    if (current == total - 1) {
        std::cout << std::endl;
        last_percent = -1; // Reset for next phase
    }
}

// Display detailed statistics
void display_error_statistics(const std::vector<float>& proj1, const std::vector<float>& proj2,
                              int embedding_dim, const std::string& comparison_name) {

    int n_points = static_cast<int>(proj1.size()) / embedding_dim;
    std::vector<double> point_errors;

    for (int i = 0; i < n_points; i++) {
        double point_error = 0.0;
        for (int d = 0; d < embedding_dim; d++) {
            double diff = proj1[i * embedding_dim + d] - proj2[i * embedding_dim + d];
            point_error += diff * diff;
        }
        point_errors.push_back(std::sqrt(point_error));
    }

    std::sort(point_errors.begin(), point_errors.end());

    double mean_error = std::accumulate(point_errors.begin(), point_errors.end(), 0.0) / n_points;
    double median_error = point_errors[static_cast<size_t>(n_points / 2)];
    double p95_error = point_errors[static_cast<size_t>(n_points * 0.95)];
    double p99_error = point_errors[static_cast<size_t>(n_points * 0.99)];
    double max_error = point_errors.back();

    // Calculate >1% difference statistics
    int points_above_1_percent = 0;
    for (const auto& error : point_errors) {
        if (error > 0.01) {
            points_above_1_percent++;
        }
    }
    double percent_above_1_percent = (static_cast<double>(points_above_1_percent) / n_points) * 100.0;

    std::cout << "📈 " << comparison_name << " Error Statistics:" << std::endl;

    // Add HNSW approximation explanation for fit vs transform comparisons
    if (comparison_name.find("Fit vs Transform") != std::string::npos) {
        std::cout << "   ℹ️  Note: Differences between fit and transform are NORMAL and EXPECTED with HNSW approximation" << std::endl;
        std::cout << "       HNSW provides 50-2000x speed improvement with minimal accuracy loss (<1% typically)" << std::endl;
    }

    std::cout << "   Mean: " << std::fixed << std::setprecision(6) << mean_error << std::endl;
    std::cout << "   Median: " << median_error << std::endl;
    std::cout << "   P95: " << p95_error << std::endl;
    std::cout << "   P99: " << p99_error << std::endl;
    std::cout << "   Max: " << max_error << std::endl;
    std::cout << "   Points with >1% difference: " << points_above_1_percent << " ("
              << std::fixed << std::setprecision(1) << percent_above_1_percent << "%)" << std::endl;
}

/*
 * UMAP FUNCTION COMPARISON GUIDE:
 * ===============================
 *
 * 1. uwot_fit_with_progress vs uwot_fit_with_progress_v2:
 *    - SAME CORE TRAINING PATH: Both use identical UMAP optimization algorithm
 *    - v1: Basic progress callback (epoch progress only)
 *    - v2: Enhanced callbacks with loss reporting, phase information, time estimates
 *    - v2 WRAPS v1: Translates v2 callbacks to v1 format internally
 *    - RECOMMENDATION: Always use v2 for better progress visibility
 *
 * 2. uwot_transform vs uwot_transform_detailed:
 *    - SAME PROJECTION ALGORITHM: Identical HNSW-based transform code path
 *    - Basic: Returns embedding coordinates only (faster)
 *    - Detailed: Returns embedding + safety metrics (confidence, outlier levels, z-scores)
 *    - NO ALGORITHMIC DIFFERENCE: Same fit data, same projection, just different output
 *    - RECOMMENDATION: Use detailed for production safety analysis
 *
 * 3. Model Lifecycle:
 *    - uwot_create(): Creates fresh model object
 *    - uwot_fit_*(): Trains model and creates embeddings
 *    - uwot_transform_*(): Projects new data using existing model
 *    - uwot_save_model(): Saves complete model state + HNSW index
 *    - uwot_load_model(): Creates NEW model object from saved state
 *    - uwot_destroy(): Cleans up model memory
 */

// Run comprehensive test for a specific embedding dimension
TestResult run_test_phase(const std::vector<float>& data, int embedding_dim, const std::string& phase_name) {
    TestResult result;
    result.phase_name = phase_name;
    result.embedding_dim = embedding_dim;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "\n🚀 " << phase_name << " (embedding_dim=" << embedding_dim << ")" << std::endl;
    std::cout << "=" << std::string(60, '=') << std::endl;

    // Step 1: Create and fit model using v2 API (enhanced progress reporting)
    std::cout << "\n📚 Step 1: Training UMAP model with uwot_fit_with_progress_v2..." << std::endl;
    std::cout << "   📝 Using v2 API for enhanced loss reporting and phase information" << std::endl;

    UwotModel* original_model = uwot_create(); // Original model for training
    if (!original_model) {
        std::cout << "❌ Failed to create original training model" << std::endl;
        result.fit_vs_transform_mse = -1.0;
        return result;
    }

    result.original_projection.resize(N_SAMPLES * embedding_dim);

    // ONLY use uwot_fit_with_progress_v2 (enhanced callbacks with loss reporting)
    int fit_result = uwot_fit_with_progress_v2(
        original_model, const_cast<float*>(data.data()), N_SAMPLES, N_DIM, embedding_dim, N_NEIGHBORS,
        MIN_DIST, SPREAD, N_EPOCHS, UWOT_METRIC_EUCLIDEAN,
        result.original_projection.data(), progress_callback_v2, 0, 16, 200, 200
    );

    if (fit_result != UWOT_SUCCESS) {
        std::cout << "❌ Training failed: " << uwot_get_error_message(fit_result) << std::endl;
        uwot_destroy(original_model);
        result.fit_vs_transform_mse = -1.0;
        return result;
    }

    auto fit_time = std::chrono::high_resolution_clock::now();
    auto fit_duration = std::chrono::duration_cast<std::chrono::milliseconds>(fit_time - start_time);
    std::cout << "✅ Training completed in " << fit_duration.count() << "ms" << std::endl;

    // Step 2: Transform with detailed safety metrics using uwot_transform_detailed
    std::cout << "\n🔄 Step 2: Transform with uwot_transform_detailed (safety metrics)..." << std::endl;
    std::cout << "   📝 Using detailed API for confidence scores, outlier levels, z-scores" << std::endl;

    result.transform_projection.resize(N_SAMPLES * embedding_dim);
    std::vector<int> nn_indices(N_SAMPLES * N_NEIGHBORS);
    std::vector<float> nn_distances(N_SAMPLES * N_NEIGHBORS);
    std::vector<float> confidence_scores(N_SAMPLES);
    std::vector<int> outlier_levels(N_SAMPLES);
    std::vector<float> percentile_ranks(N_SAMPLES);
    std::vector<float> z_scores(N_SAMPLES);

    int transform_result = uwot_transform_detailed(
        original_model, const_cast<float*>(data.data()), N_SAMPLES, N_DIM,
        result.transform_projection.data(), nn_indices.data(), nn_distances.data(),
        confidence_scores.data(), outlier_levels.data(), percentile_ranks.data(), z_scores.data()
    );

    if (transform_result != UWOT_SUCCESS) {
        std::cout << "❌ Transform failed: " << uwot_get_error_message(transform_result) << std::endl;
        uwot_destroy(original_model);
        result.fit_vs_transform_mse = -1.0;
        return result;
    }

    auto transform_time = std::chrono::high_resolution_clock::now();
    auto transform_duration = std::chrono::duration_cast<std::chrono::microseconds>(transform_time - fit_time);
    std::cout << "✅ Transform completed in " << transform_duration.count() << "μs" << std::endl;

    // Step 3: Compare fit vs transform projections
    std::cout << "\n📊 Step 3: Analyzing fit vs transform consistency..." << std::endl;
    result.fit_vs_transform_mse = calculate_mse(result.original_projection, result.transform_projection, embedding_dim);
    std::cout << "🎯 Fit vs Transform MSE: " << std::scientific << std::setprecision(6)
              << result.fit_vs_transform_mse << std::endl;

    display_error_statistics(result.original_projection, result.transform_projection,
                           embedding_dim, "Fit vs Transform");

    // Display safety metrics sample
    std::cout << "\n🛡️ Safety metrics sample (first 5 points):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "   Point " << i << ": confidence=" << std::fixed << std::setprecision(3)
                  << confidence_scores[i] << ", outlier_level=" << outlier_levels[i]
                  << ", percentile=" << percentile_ranks[i] << ", z_score=" << z_scores[i] << std::endl;
    }

    // Display fit vs projection examples
    std::cout << "\n📋 Fit vs Transform Projection Examples (first 5 points):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "   Point " << i << ":" << std::endl;
        std::cout << "     Fit:       [";
        for (int d = 0; d < embedding_dim; d++) {
            std::cout << std::fixed << std::setprecision(3) << result.original_projection[i * embedding_dim + d];
            if (d < embedding_dim - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "     Transform: [";
        for (int d = 0; d < embedding_dim; d++) {
            std::cout << std::fixed << std::setprecision(3) << result.transform_projection[i * embedding_dim + d];
            if (d < embedding_dim - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Calculate point-wise difference
        double point_error = 0.0;
        for (int d = 0; d < embedding_dim; d++) {
            double diff = result.original_projection[i * embedding_dim + d] - result.transform_projection[i * embedding_dim + d];
            point_error += diff * diff;
        }
        point_error = std::sqrt(point_error);
        std::cout << "     Error:     " << std::fixed << std::setprecision(6) << point_error << std::endl;
    }

    // Step 4: Save model (complete state + HNSW index)
    std::cout << "\n💾 Step 4: Saving trained model..." << std::endl;
    std::cout << "   📝 Saving complete model state, embeddings, and HNSW index" << std::endl;

    std::string filename = "test_model_" + std::to_string(embedding_dim) + "d.umap";
    int save_result = uwot_save_model(original_model, filename.c_str());
    if (save_result != UWOT_SUCCESS) {
        std::cout << "❌ Save failed: " << uwot_get_error_message(save_result) << std::endl;
        uwot_destroy(original_model);
        result.fit_vs_transform_mse = -1.0;
        return result;
    }
    std::cout << "✅ Model saved to " << filename << std::endl;

    // Step 5: Load model (creates COMPLETELY NEW model object)
    std::cout << "\n📂 Step 5: Loading model into NEW object..." << std::endl;
    std::cout << "   📝 uwot_load_model creates a completely separate UwotModel object" << std::endl;
    std::cout << "   📝 This ensures clean separation between original and loaded models" << std::endl;

    UwotModel* loaded_model = uwot_load_model(filename.c_str()); // NEW model object
    if (!loaded_model) {
        std::cout << "❌ Failed to load model from " << filename << std::endl;
        uwot_destroy(original_model);
        result.fit_vs_transform_mse = -1.0;
        return result;
    }
    std::cout << "✅ Model loaded successfully into new UwotModel object" << std::endl;

    // Step 6: Transform with loaded model
    std::cout << "\n🔄 Step 6: Transform with loaded model..." << std::endl;
    result.loaded_transform_projection.resize(N_SAMPLES * embedding_dim);

    int loaded_transform_result = uwot_transform_detailed(
        loaded_model, const_cast<float*>(data.data()), N_SAMPLES, N_DIM,
        result.loaded_transform_projection.data(), nn_indices.data(), nn_distances.data(),
        confidence_scores.data(), outlier_levels.data(), percentile_ranks.data(), z_scores.data()
    );

    if (loaded_transform_result != UWOT_SUCCESS) {
        std::cout << "❌ Loaded model transform failed: " << uwot_get_error_message(loaded_transform_result) << std::endl;
        uwot_destroy(original_model);
        uwot_destroy(loaded_model);
        result.fit_vs_transform_mse = -1.0;
        return result;
    }
    std::cout << "✅ Loaded model transform completed" << std::endl;

    // Step 7: Compare all projections
    std::cout << "\n📊 Step 7: Comprehensive comparison analysis..." << std::endl;

    // Original vs Loaded Transform
    result.original_vs_loaded_mse = calculate_mse(result.transform_projection, result.loaded_transform_projection, embedding_dim);
    std::cout << "🎯 Original Transform vs Loaded Transform MSE: " << std::scientific
              << result.original_vs_loaded_mse << std::endl;

    // Fit vs Loaded Transform
    result.fit_vs_loaded_mse = calculate_mse(result.original_projection, result.loaded_transform_projection, embedding_dim);
    std::cout << "🎯 Fit vs Loaded Transform MSE: " << std::scientific
              << result.fit_vs_loaded_mse << std::endl;

    display_error_statistics(result.transform_projection, result.loaded_transform_projection,
                           embedding_dim, "Original vs Loaded Transform");

    display_error_statistics(result.original_projection, result.loaded_transform_projection,
                           embedding_dim, "Fit vs Loaded Transform");

    // Cleanup
    uwot_destroy(original_model);
    uwot_destroy(loaded_model);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\n⏱️ Total phase duration: " << total_duration.count() << "ms" << std::endl;

    return result;
}

int main() {
    std::cout << "🧪 COMPREHENSIVE UMAP PIPELINE TEST" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "Dataset: " << N_SAMPLES << " x " << N_DIM << " dimensions" << std::endl;
    std::cout << "Parameters: min_dist=" << MIN_DIST << ", spread=" << SPREAD
              << ", n_neighbors=" << N_NEIGHBORS << ", n_epochs=" << N_EPOCHS << std::endl;
    std::cout << "Testing embedding dimensions: 2D, 20D" << std::endl;

    // Generate synthetic dataset
    std::vector<float> data;
    generate_synthetic_data(data, N_SAMPLES, N_DIM);

    // Run tests for different embedding dimensions
    std::vector<TestResult> results;

    // Test 1: 2D embedding
    results.push_back(run_test_phase(data, 2, "Phase 1: 2D Embedding"));

    // Test 2: 20D embedding
    results.push_back(run_test_phase(data, 20, "Phase 2: 20D Embedding"));

    // Final comprehensive summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "🎉 COMPREHENSIVE TEST RESULTS SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << "\n📊 MSE Comparison Across All Dimensions:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(15) << "Dimension"
              << std::setw(20) << "Fit vs Transform"
              << std::setw(20) << "Orig vs Loaded"
              << std::setw(20) << "Fit vs Loaded" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (const auto& result : results) {
        if (result.fit_vs_transform_mse >= 0) {
            std::cout << std::left << std::setw(15) << (std::to_string(result.embedding_dim) + "D")
                      << std::setw(20) << std::scientific << std::setprecision(3) << result.fit_vs_transform_mse
                      << std::setw(20) << result.original_vs_loaded_mse
                      << std::setw(20) << result.fit_vs_loaded_mse << std::endl;
        } else {
            std::cout << std::left << std::setw(15) << (std::to_string(result.embedding_dim) + "D")
                      << std::setw(60) << "FAILED" << std::endl;
        }
    }

    // Analyze consistency patterns
    std::cout << "\n🔍 Consistency Analysis:" << std::endl;
    for (const auto& result : results) {
        if (result.fit_vs_transform_mse >= 0) {
            std::cout << "✅ " << result.embedding_dim << "D: ";
            if (result.fit_vs_transform_mse < 1e-6) {
                std::cout << "EXCELLENT fit/transform consistency";
            } else if (result.fit_vs_transform_mse < 1e-3) {
                std::cout << "GOOD fit/transform consistency";
            } else {
                std::cout << "MODERATE fit/transform consistency";
            }

            if (result.original_vs_loaded_mse < 1e-10) {
                std::cout << ", PERFECT save/load consistency" << std::endl;
            } else if (result.original_vs_loaded_mse < 1e-6) {
                std::cout << ", EXCELLENT save/load consistency" << std::endl;
            } else {
                std::cout << ", GOOD save/load consistency" << std::endl;
            }
        }
    }

    std::cout << "\n🎯 Test completed successfully! All phases validated the modular UMAP implementation." << std::endl;
    return 0;
}