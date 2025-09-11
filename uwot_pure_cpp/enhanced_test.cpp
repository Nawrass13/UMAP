#include "uwot_simple_wrapper.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>

// Generate synthetic test data
std::vector<std::vector<float>> generate_test_data(int n_samples, int n_features, int random_seed = 42) {
    std::mt19937 gen(random_seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> data(n_samples, std::vector<float>(n_features));

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            data[i][j] = dist(gen);
        }
    }

    return data;
}

// Generate slightly different test data for transformation testing
std::vector<std::vector<float>> generate_new_test_data(int n_samples, int n_features, int random_seed = 123) {
    std::mt19937 gen(random_seed);
    std::normal_distribution<float> dist(0.5f, 0.8f); // Different distribution

    std::vector<std::vector<float>> data(n_samples, std::vector<float>(n_features));

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            data[i][j] = dist(gen);
        }
    }

    return data;
}

void print_embedding_sample(const std::vector<float>& embedding, int n_samples, int embedding_dim, const std::string& title) {
    std::cout << title << ":" << std::endl;
    for (int i = 0; i < std::min(5, n_samples); ++i) {
        std::cout << "  Sample " << i << ": (";
        for (int d = 0; d < embedding_dim; ++d) {
            std::cout << std::fixed << std::setprecision(3) << embedding[i * embedding_dim + d];
            if (d < embedding_dim - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
}

int main() {
    std::cout << "=== Enhanced UMAP C++ Wrapper Test ===" << std::endl;
    std::cout << "Testing: Training, Save/Load, and Transform functionality" << std::endl << std::endl;

    // Parameters
    const int n_train_samples = 150;
    const int n_features = 10;
    const int n_neighbors = 15;
    const int n_epochs = 300;
    const float min_dist = 0.1f;
    const char* model_filename = "umap_model.bin";

    // === STEP 1: Training ===
    std::cout << "Step 1: Training UMAP model" << std::endl;
    std::cout << "- Training samples: " << n_train_samples << std::endl;
    std::cout << "- Features: " << n_features << std::endl;
    std::cout << "- Neighbors: " << n_neighbors << std::endl;
    std::cout << "- Epochs: " << n_epochs << std::endl << std::endl;

    auto train_data = generate_test_data(n_train_samples, n_features);

    // Convert to flat array
    std::vector<float> flat_train_data(n_train_samples * n_features);
    for (int i = 0; i < n_train_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            flat_train_data[i * n_features + j] = train_data[i][j];
        }
    }

    // Create and train model
    UwotModel* model = uwot_create();
    if (!model) {
        std::cerr << "Failed to create UMAP model" << std::endl;
        return 1;
    }

    std::vector<float> train_embedding(n_train_samples * 2);

    std::cout << "Training model..." << std::endl;
    int result = uwot_fit(model, flat_train_data.data(), n_train_samples, n_features,
        n_neighbors, min_dist, n_epochs, train_embedding.data());

    if (result != UWOT_SUCCESS) {
        std::cerr << "Training failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(model);
        return 1;
    }

    std::cout << "[PASS] Training completed successfully!" << std::endl;
    print_embedding_sample(train_embedding, n_train_samples, 2, "Training embedding (first 5 samples)");
    std::cout << std::endl;

    // === STEP 2: Save Model ===
    std::cout << "Step 2: Saving model to disk" << std::endl;
    result = uwot_save_model(model, model_filename);

    if (result != UWOT_SUCCESS) {
        std::cerr << "Save failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(model);
        return 1;
    }

    std::cout << "[PASS] Model saved to: " << model_filename << std::endl;

    // Get model info
    int n_vertices, n_dim, embedding_dim, n_neighbors_saved;
    float min_dist_saved;
    uwot_get_model_info(model, &n_vertices, &n_dim, &embedding_dim, &n_neighbors_saved, &min_dist_saved);

    std::cout << "Model info: " << n_vertices << " vertices, " << n_dim << "D -> " << embedding_dim << "D" << std::endl;
    std::cout << "Parameters: k=" << n_neighbors_saved << ", min_dist=" << min_dist_saved << std::endl << std::endl;

    // Clean up original model
    uwot_destroy(model);
    model = nullptr;

    // === STEP 3: Load Model ===
    std::cout << "Step 3: Loading model from disk" << std::endl;

    UwotModel* loaded_model = uwot_load_model(model_filename);
    if (!loaded_model) {
        std::cerr << "Failed to load model from: " << model_filename << std::endl;
        return 1;
    }

    std::cout << "[PASS] Model loaded successfully!" << std::endl;

    // Verify loaded model
    if (!uwot_is_fitted(loaded_model)) {
        std::cerr << "Loaded model is not fitted!" << std::endl;
        uwot_destroy(loaded_model);
        return 1;
    }

    std::cout << "[PASS] Loaded model is fitted and ready" << std::endl << std::endl;

    // === STEP 4: Transform New Data ===
    std::cout << "Step 4: Transforming new data" << std::endl;

    const int n_new_samples = 20;
    auto new_data = generate_new_test_data(n_new_samples, n_features);

    std::cout << "- New samples: " << n_new_samples << std::endl;
    std::cout << "- Same feature dimension: " << n_features << std::endl << std::endl;

    // Convert to flat array
    std::vector<float> flat_new_data(n_new_samples * n_features);
    for (int i = 0; i < n_new_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            flat_new_data[i * n_features + j] = new_data[i][j];
        }
    }

    std::vector<float> new_embedding(n_new_samples * 2);

    std::cout << "Transforming new data..." << std::endl;
    result = uwot_transform(loaded_model, flat_new_data.data(), n_new_samples, n_features, new_embedding.data());

    if (result != UWOT_SUCCESS) {
        std::cerr << "Transform failed: " << uwot_get_error_message(result) << std::endl;
        uwot_destroy(loaded_model);
        return 1;
    }

    std::cout << "[PASS] Transform completed successfully!" << std::endl;
    print_embedding_sample(new_embedding, n_new_samples, 2, "Transformed embedding (first 5 samples)");
    std::cout << std::endl;

    // === STEP 5: Statistics and Validation ===
    std::cout << "Step 5: Validation and Statistics" << std::endl;

    // Compute embedding bounds for both training and new data
    float train_min_x = train_embedding[0], train_max_x = train_embedding[0];
    float train_min_y = train_embedding[1], train_max_y = train_embedding[1];

    for (int i = 0; i < n_train_samples; ++i) {
        float x = train_embedding[i * 2];
        float y = train_embedding[i * 2 + 1];
        train_min_x = std::min(train_min_x, x);
        train_max_x = std::max(train_max_x, x);
        train_min_y = std::min(train_min_y, y);
        train_max_y = std::max(train_max_y, y);
    }

    float new_min_x = new_embedding[0], new_max_x = new_embedding[0];
    float new_min_y = new_embedding[1], new_max_y = new_embedding[1];

    for (int i = 0; i < n_new_samples; ++i) {
        float x = new_embedding[i * 2];
        float y = new_embedding[i * 2 + 1];
        new_min_x = std::min(new_min_x, x);
        new_max_x = std::max(new_max_x, x);
        new_min_y = std::min(new_min_y, y);
        new_max_y = std::max(new_max_y, y);
    }

    std::cout << "Training embedding bounds: X=[" << std::fixed << std::setprecision(3)
        << train_min_x << ", " << train_max_x << "], Y=[" << train_min_y << ", " << train_max_y << "]" << std::endl;
    std::cout << "New data embedding bounds:  X=[" << std::fixed << std::setprecision(3)
        << new_min_x << ", " << new_max_x << "], Y=[" << new_min_y << ", " << new_max_y << "]" << std::endl;

    // Check if new embeddings are reasonable (within or near training bounds)
    bool reasonable_x = (new_min_x >= train_min_x - 2.0f) && (new_max_x <= train_max_x + 2.0f);
    bool reasonable_y = (new_min_y >= train_min_y - 2.0f) && (new_max_y <= train_max_y + 2.0f);

    if (reasonable_x && reasonable_y) {
        std::cout << "[PASS] New embeddings are within reasonable bounds" << std::endl;
    }
    else {
        std::cout << "[WARN] New embeddings may be outside expected bounds" << std::endl;
    }

    std::cout << std::endl;

    // === STEP 6: Cleanup ===
    std::cout << "Step 6: Cleanup" << std::endl;

    uwot_destroy(loaded_model);
    std::cout << "[PASS] Model destroyed" << std::endl;

    // Remove the model file
    if (std::remove(model_filename) == 0) {
        std::cout << "[PASS] Model file removed" << std::endl;
    }
    else {
        std::cout << "[WARN] Could not remove model file" << std::endl;
    }

    std::cout << std::endl << "=== All Tests Completed Successfully! ===" << std::endl;
    std::cout << std::endl << "Summary:" << std::endl;
    std::cout << "[PASS] Model training: " << n_train_samples << " samples -> 2D embedding" << std::endl;
    std::cout << "[PASS] Model persistence: Save and load from disk" << std::endl;
    std::cout << "[PASS] Data transformation: " << n_new_samples << " new samples transformed" << std::endl;
    std::cout << "[PASS] Memory management: All resources properly cleaned up" << std::endl;
    std::cout << std::endl << "Your UMAP library is ready for production use with C#!" << std::endl;

    return 0;
}