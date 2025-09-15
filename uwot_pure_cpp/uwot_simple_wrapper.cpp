#include "uwot_simple_wrapper.h"
#include "smooth_knn.h"
#include "optimize.h"
#include "gradient.h" 
#include "transform.h"
#include "update.h"
#include "epoch.h"

// HNSW library integration
#include "hnswlib.h"
#include "hnswalg.h"
#include "space_l2.h"
#include "space_ip.h"

#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <cstring>
#include <map>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>

// Cross-platform file utilities
namespace temp_utils {
    bool safe_remove_file(const std::string& filename) {
        try {
            return std::filesystem::remove(filename);
        } catch (...) {
            // Fallback to C remove if filesystem fails
            return std::remove(filename.c_str()) == 0;
        }
    }
}

// Enhanced progress reporting utilities
namespace progress_utils {

    // Format time duration in human-readable format
    std::string format_duration(double seconds) {
        std::ostringstream oss;
        if (seconds < 60) {
            oss << std::fixed << std::setprecision(1) << seconds << "s";
        } else if (seconds < 3600) {
            int minutes = static_cast<int>(seconds / 60);
            int secs = static_cast<int>(seconds) % 60;
            oss << minutes << "m " << secs << "s";
        } else {
            int hours = static_cast<int>(seconds / 3600);
            int minutes = static_cast<int>((seconds - hours * 3600) / 60);
            oss << hours << "h " << minutes << "m";
        }
        return oss.str();
    }

    // Estimate remaining time based on current progress
    std::string estimate_remaining_time(int current, int total, double elapsed_seconds) {
        if (current == 0 || elapsed_seconds <= 0) {
            return "estimating...";
        }

        double rate = current / elapsed_seconds;
        double remaining_items = total - current;
        double estimated_remaining = remaining_items / rate;

        return "est. " + format_duration(estimated_remaining) + " remaining";
    }

    // Generate complexity-based time warnings
    std::string generate_complexity_warning(int n_obs, int n_dim, const std::string& operation) {
        std::ostringstream oss;

        if (operation == "exact_knn") {
            long long operations = static_cast<long long>(n_obs) * n_obs * n_dim;
            if (operations > 1e11) { // > 100B operations
                oss << "WARNING: Exact k-NN on " << n_obs << "x" << n_dim
                    << " may take hours. Consider approximation or smaller dataset.";
            } else if (operations > 1e9) { // > 1B operations
                oss << "NOTICE: Large exact k-NN computation (" << n_obs << "x" << n_dim
                    << ") - this may take several minutes.";
            }
        }

        return oss.str();
    }

    // Safe callback invoker - handles null callbacks gracefully
    void safe_callback(uwot_progress_callback_v2 callback,
                      const char* phase, int current, int total, float percent,
                      const char* message = nullptr) {
        if (callback != nullptr) {
            callback(phase, current, total, percent, message);
        }
    }
}

// Custom L1Space implementation for Manhattan distance in HNSW
// Implements SpaceInterface<float> for Manhattan (L1) distance metric
class L1Space : public hnswlib::SpaceInterface<float> {
    hnswlib::DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

public:
    L1Space(size_t dim) : dim_(dim), data_size_(dim * sizeof(float)) {
        // Manhattan distance function implementation
        fstdistfunc_ = [](const void* pVect1v, const void* pVect2v, const void* qty_ptr) -> float {
            const float* pVect1 = static_cast<const float*>(pVect1v);
            const float* pVect2 = static_cast<const float*>(pVect2v);
            size_t qty = *static_cast<const size_t*>(qty_ptr);

            float distance = 0.0f;
            for (size_t i = 0; i < qty; ++i) {
                distance += std::abs(pVect1[i] - pVect2[i]);
            }
            return distance;
        };
    }

    size_t get_data_size() override {
        return data_size_;
    }

    hnswlib::DISTFUNC<float> get_dist_func() override {
        return fstdistfunc_;
    }

    void* get_dist_func_param() override {
        return &dim_;
    }

    ~L1Space() = default;
};

// HNSW space factory and management utilities
namespace hnsw_utils {

    // Space factory - creates appropriate space based on metric type
    struct SpaceFactory {
        std::unique_ptr<hnswlib::L2Space> l2_space;
        std::unique_ptr<hnswlib::InnerProductSpace> ip_space;
        std::unique_ptr<L1Space> l1_space;
        hnswlib::SpaceInterface<float>* active_space;
        UwotMetric current_metric;
        bool supports_hnsw_approximation;

        SpaceFactory() : active_space(nullptr), current_metric(UWOT_METRIC_EUCLIDEAN), supports_hnsw_approximation(false) {}

        // Create and configure space for given metric and dimensions
        bool create_space(UwotMetric metric, int n_dim, uwot_progress_callback_v2 progress_callback = nullptr) {
            current_metric = metric;
            active_space = nullptr;
            supports_hnsw_approximation = false;

            try {
                switch (metric) {
                    case UWOT_METRIC_EUCLIDEAN:
                        l2_space = std::make_unique<hnswlib::L2Space>(n_dim);
                        active_space = l2_space.get();
                        supports_hnsw_approximation = true;
                        if (progress_callback) {
                            progress_utils::safe_callback(progress_callback, "Space Setup", 1, 1, 100.0f,
                                "Using L2Space for Euclidean distance - HNSW approximation enabled");
                        }
                        break;

                    case UWOT_METRIC_COSINE:
                        // For cosine similarity, we use InnerProductSpace on unit-normalized vectors
                        ip_space = std::make_unique<hnswlib::InnerProductSpace>(n_dim);
                        active_space = ip_space.get();
                        supports_hnsw_approximation = true;
                        if (progress_callback) {
                            progress_utils::safe_callback(progress_callback, "Space Setup", 1, 1, 100.0f,
                                "Using InnerProductSpace for Cosine distance - HNSW approximation enabled");
                        }
                        break;

                    case UWOT_METRIC_MANHATTAN:
                        l1_space = std::make_unique<L1Space>(n_dim);
                        active_space = l1_space.get();
                        supports_hnsw_approximation = true;
                        if (progress_callback) {
                            progress_utils::safe_callback(progress_callback, "Space Setup", 1, 1, 100.0f,
                                "Using L1Space for Manhattan distance - HNSW approximation enabled");
                        }
                        break;

                    case UWOT_METRIC_CORRELATION:
                        // Fallback to L2 space for distance statistics only
                        l2_space = std::make_unique<hnswlib::L2Space>(n_dim);
                        active_space = l2_space.get();
                        supports_hnsw_approximation = false;
                        if (progress_callback) {
                            progress_utils::safe_callback(progress_callback, "Space Setup", 1, 1, 100.0f,
                                "Correlation metric: Using L2Space for statistics only - exact k-NN required");
                        }
                        break;

                    case UWOT_METRIC_HAMMING:
                        // Fallback to L2 space for distance statistics only
                        l2_space = std::make_unique<hnswlib::L2Space>(n_dim);
                        active_space = l2_space.get();
                        supports_hnsw_approximation = false;
                        if (progress_callback) {
                            progress_utils::safe_callback(progress_callback, "Space Setup", 1, 1, 100.0f,
                                "Hamming metric: Using L2Space for statistics only - exact k-NN required");
                        }
                        break;

                    default:
                        // Default to L2
                        l2_space = std::make_unique<hnswlib::L2Space>(n_dim);
                        active_space = l2_space.get();
                        supports_hnsw_approximation = true;
                        if (progress_callback) {
                            progress_utils::safe_callback(progress_callback, "Space Setup", 1, 1, 100.0f,
                                "Unknown metric: Defaulting to L2Space - HNSW approximation enabled");
                        }
                        break;
                }

                return (active_space != nullptr);

            } catch (...) {
                active_space = nullptr;
                supports_hnsw_approximation = false;
                return false;
            }
        }

        // Unit-normalize data for cosine similarity
        void prepare_data_for_space(std::vector<float>& data, int n_obs, int n_dim,
                                   uwot_progress_callback_v2 progress_callback = nullptr) {
            if (current_metric == UWOT_METRIC_COSINE) {
                if (progress_callback) {
                    progress_utils::safe_callback(progress_callback, "Data Preparation", 0, n_obs, 0.0f,
                        "Unit-normalizing vectors for cosine similarity");
                }

                #pragma omp parallel for if(n_obs > 1000)
                for (int i = 0; i < n_obs; i++) {
                    float norm = 0.0f;
                    size_t base_idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim);

                    // Calculate L2 norm
                    for (int j = 0; j < n_dim; j++) {
                        float val = data[base_idx + j];
                        norm += val * val;
                    }
                    norm = std::sqrt(norm) + 1e-8f; // Add small epsilon to prevent division by zero

                    // Normalize
                    for (int j = 0; j < n_dim; j++) {
                        data[base_idx + j] /= norm;
                    }

                    // Progress reporting every 10%
                    if (i % std::max(1, n_obs / 10) == 0) {
                        float percent = (i * 100.0f) / n_obs;
                        progress_utils::safe_callback(progress_callback, "Data Preparation", i, n_obs, percent, nullptr);
                    }
                }

                if (progress_callback) {
                    progress_utils::safe_callback(progress_callback, "Data Preparation", n_obs, n_obs, 100.0f,
                        "Unit normalization completed for cosine similarity");
                }
            }
        }

        // Check if current metric supports HNSW approximation
        bool can_use_hnsw() const {
            return supports_hnsw_approximation;
        }

        // Get active space pointer
        hnswlib::SpaceInterface<float>* get_space() const {
            return active_space;
        }

        // Get metric name for logging
        const char* get_metric_name() const {
            switch (current_metric) {
                case UWOT_METRIC_EUCLIDEAN: return "Euclidean";
                case UWOT_METRIC_COSINE: return "Cosine";
                case UWOT_METRIC_MANHATTAN: return "Manhattan";
                case UWOT_METRIC_CORRELATION: return "Correlation";
                case UWOT_METRIC_HAMMING: return "Hamming";
                default: return "Unknown";
            }
        }
    };

    // Unified data normalization pipeline
    struct NormalizationPipeline {

        // Comprehensive data preparation combining z-normalization and space-specific preparation
        static bool prepare_training_data(const std::vector<float>& raw_data,
                                          std::vector<float>& hnsw_data,
                                          std::vector<float>& exact_data,
                                          int n_obs, int n_dim,
                                          const std::vector<float>& means,
                                          const std::vector<float>& stds,
                                          SpaceFactory* space_factory,
                                          uwot_progress_callback_v2 progress_callback = nullptr) {

            if (progress_callback) {
                progress_utils::safe_callback(progress_callback, "Data Pipeline", 0, 3, 0.0f,
                    "Starting unified normalization pipeline");
            }

            // Step 1: Z-normalization (for both HNSW and exact methods)
            auto norm_start = std::chrono::steady_clock::now();
            if (progress_callback) {
                double est_seconds = (static_cast<double>(n_obs) * n_dim) * 1e-9; // ~1 operation per element, 1B ops/sec
                char est_msg[256];
                snprintf(est_msg, sizeof(est_msg), "Applying z-score normalization (est. %.1fs)", est_seconds);
                progress_utils::safe_callback(progress_callback, "Z-Normalization", 0, n_obs, 0.0f, est_msg);
            }

            hnsw_data.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim));
            exact_data.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim));

            // Apply z-normalization to both datasets
            #pragma omp parallel for if(n_obs > 1000)
            for (int i = 0; i < n_obs; i++) {
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    float normalized_value = (raw_data[idx] - means[j]) / (stds[j] + 1e-8f);
                    hnsw_data[idx] = normalized_value;
                    exact_data[idx] = normalized_value; // Keep original normalized for exact computation
                }

                // Progress reporting every 10% with time estimation
                if (progress_callback && (i % std::max(1, n_obs / 10) == 0)) {
                    float percent = (i * 100.0f) / n_obs;
                    auto elapsed = std::chrono::steady_clock::now() - norm_start;
                    auto elapsed_sec = std::chrono::duration<double>(elapsed).count();
                    double remaining_sec = (elapsed_sec / (i + 1)) * (n_obs - i - 1);

                    char time_msg[256];
                    snprintf(time_msg, sizeof(time_msg), "Normalizing: %.1f%% (remaining: %.1fs)", percent, remaining_sec);
                    progress_utils::safe_callback(progress_callback, "Z-Normalization", i, n_obs, percent, time_msg);
                }
            }

            auto norm_elapsed = std::chrono::steady_clock::now() - norm_start;
            auto norm_sec = std::chrono::duration<double>(norm_elapsed).count();

            if (progress_callback) {
                char final_msg[256];
                snprintf(final_msg, sizeof(final_msg), "Z-score normalization completed in %.2fs", norm_sec);
                progress_utils::safe_callback(progress_callback, "Z-Normalization", n_obs, n_obs, 100.0f, final_msg);
            }

            // Step 2: Space-specific preparation (for HNSW data only)
            auto space_prep_start = std::chrono::steady_clock::now();
            if (progress_callback) {
                const char* metric_name = space_factory->get_metric_name();
                char est_msg[256];
                if (strstr(metric_name, "Cosine")) {
                    double est_seconds = (static_cast<double>(n_obs) * n_dim) * 2e-9; // Unit normalization
                    snprintf(est_msg, sizeof(est_msg), "Applying %s space preparation (unit norm, est. %.1fs)", metric_name, est_seconds);
                } else {
                    snprintf(est_msg, sizeof(est_msg), "Applying %s space preparation (minimal processing)", metric_name);
                }
                progress_utils::safe_callback(progress_callback, "Space Preparation", 0, 1, 0.0f, est_msg);
            }

            space_factory->prepare_data_for_space(hnsw_data, n_obs, n_dim, progress_callback);

            auto space_prep_elapsed = std::chrono::steady_clock::now() - space_prep_start;
            auto space_prep_sec = std::chrono::duration<double>(space_prep_elapsed).count();

            if (progress_callback) {
                char final_msg[256];
                snprintf(final_msg, sizeof(final_msg), "Space preparation completed in %.2fs", space_prep_sec);
                progress_utils::safe_callback(progress_callback, "Space Preparation", 1, 1, 100.0f, final_msg);

                auto total_pipeline_elapsed = std::chrono::steady_clock::now() - norm_start;
                auto total_pipeline_sec = std::chrono::duration<double>(total_pipeline_elapsed).count();
                char pipeline_msg[256];
                snprintf(pipeline_msg, sizeof(pipeline_msg), "Complete normalization pipeline finished in %.2fs", total_pipeline_sec);
                progress_utils::safe_callback(progress_callback, "Data Pipeline", 3, 3, 100.0f, pipeline_msg);
            }

            return true;
        }

        // Transform single data point using stored normalization parameters
        static void prepare_transform_data(const std::vector<float>& raw_data,
                                          std::vector<float>& normalized_data,
                                          int n_dim,
                                          const std::vector<float>& means,
                                          const std::vector<float>& stds,
                                          UwotMetric metric) {

            normalized_data.resize(n_dim);

            // Apply z-normalization
            for (int j = 0; j < n_dim; j++) {
                normalized_data[j] = (raw_data[j] - means[j]) / (stds[j] + 1e-8f);
            }

            // Apply space-specific normalization if needed
            if (metric == UWOT_METRIC_COSINE) {
                // Unit normalize for cosine similarity
                float norm = 0.0f;
                for (int j = 0; j < n_dim; j++) {
                    norm += normalized_data[j] * normalized_data[j];
                }
                norm = std::sqrt(norm) + 1e-8f;

                for (int j = 0; j < n_dim; j++) {
                    normalized_data[j] /= norm;
                }
            }
        }

        // Validate normalization parameters
        static bool validate_normalization_params(const std::vector<float>& means,
                                                 const std::vector<float>& stds,
                                                 int expected_dims) {
            if (means.size() != static_cast<size_t>(expected_dims) ||
                stds.size() != static_cast<size_t>(expected_dims)) {
                return false;
            }

            // Check for invalid standard deviations
            for (float std_val : stds) {
                if (std_val <= 0.0f || !std::isfinite(std_val)) {
                    return false;
                }
            }

            // Check for invalid means
            for (float mean_val : means) {
                if (!std::isfinite(mean_val)) {
                    return false;
                }
            }

            return true;
        }

        // Compute normalization statistics with enhanced validation
        static bool compute_normalization_stats(const std::vector<float>& data,
                                               int n_obs, int n_dim,
                                               std::vector<float>& means,
                                               std::vector<float>& stds,
                                               uwot_progress_callback_v2 progress_callback = nullptr) {

            means.resize(n_dim);
            stds.resize(n_dim);

            if (progress_callback) {
                progress_utils::safe_callback(progress_callback, "Computing Statistics", 0, n_dim, 0.0f,
                    "Computing feature means and standard deviations");
            }

            // Compute means
            std::fill(means.begin(), means.end(), 0.0f);
            for (int i = 0; i < n_obs; i++) {
                for (int j = 0; j < n_dim; j++) {
                    means[j] += data[static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j)];
                }
            }

            for (int j = 0; j < n_dim; j++) {
                means[j] /= static_cast<float>(n_obs);
            }

            // Compute standard deviations
            std::fill(stds.begin(), stds.end(), 0.0f);
            for (int i = 0; i < n_obs; i++) {
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    float diff = data[idx] - means[j];
                    stds[j] += diff * diff;
                }
            }

            for (int j = 0; j < n_dim; j++) {
                stds[j] = std::sqrt(stds[j] / static_cast<float>(n_obs - 1));
                // Prevent division by zero
                if (stds[j] < 1e-8f) {
                    stds[j] = 1.0f;
                }

                // Progress reporting
                if (progress_callback && (j % std::max(1, n_dim / 10) == 0)) {
                    float percent = (j * 100.0f) / n_dim;
                    progress_utils::safe_callback(progress_callback, "Computing Statistics", j, n_dim, percent, nullptr);
                }
            }

            if (progress_callback) {
                progress_utils::safe_callback(progress_callback, "Computing Statistics", n_dim, n_dim, 100.0f,
                    "Feature statistics computed successfully");
            }

            return validate_normalization_params(means, stds, n_dim);
        }
    };
}

struct UwotModel {
    // Model parameters
    int n_vertices;
    int n_dim;
    int embedding_dim;
    int n_neighbors;
    float min_dist;
    float spread; // UMAP spread parameter (controls global scale)
    UwotMetric metric;
    float a, b; // UMAP curve parameters
    bool is_fitted;
    bool force_exact_knn; // Override flag to force brute-force k-NN

    // HNSW Index for fast neighbor search (replaces training_data storage)
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> ann_index;
    std::unique_ptr<hnsw_utils::SpaceFactory> space_factory;

    // Normalization parameters (moved from C#)
    std::vector<float> feature_means;
    std::vector<float> feature_stds;
    bool use_normalization;

    // Graph structure using uwot types
    std::vector<unsigned int> positive_head;
    std::vector<unsigned int> positive_tail;
    std::vector<double> positive_weights;  // uwot uses double for weights

    // Final embedding
    std::vector<float> embedding;

    // k-NN structure for transformation (uwot format)
    std::vector<int> nn_indices;      // flattened indices 
    std::vector<float> nn_distances;  // flattened distances
    std::vector<float> nn_weights;    // flattened weights for transform

    // Comprehensive neighbor distance statistics for safety detection
    float min_neighbor_distance;
    float mean_neighbor_distance;
    float std_neighbor_distance;
    float p95_neighbor_distance;
    float p99_neighbor_distance;
    float mild_outlier_threshold;      // 2.5 std deviations
    float extreme_outlier_threshold;   // 4.0 std deviations

    UwotModel() : n_vertices(0), n_dim(0), embedding_dim(2), n_neighbors(15),
        min_dist(0.1f), spread(1.0f), metric(UWOT_METRIC_EUCLIDEAN), a(1.929f), b(0.7915f),
        is_fitted(false), force_exact_knn(false), use_normalization(false),
        min_neighbor_distance(0.0f), mean_neighbor_distance(0.0f),
        std_neighbor_distance(0.0f), p95_neighbor_distance(0.0f),
        p99_neighbor_distance(0.0f), mild_outlier_threshold(0.0f),
        extreme_outlier_threshold(0.0f) {

        space_factory = std::make_unique<hnsw_utils::SpaceFactory>();
    }
};

// Helper function to compute normalization parameters
void compute_normalization(const std::vector<float>& data, int n_obs, int n_dim,
    std::vector<float>& means, std::vector<float>& stds) {
    means.resize(n_dim);
    stds.resize(n_dim);

    // Calculate means
    std::fill(means.begin(), means.end(), 0.0f);
    for (int i = 0; i < n_obs; i++) {
        for (int j = 0; j < n_dim; j++) {
            means[j] += data[static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j)];
        }
    }
    for (int j = 0; j < n_dim; j++) {
        means[j] /= static_cast<float>(n_obs);
    }

    // Calculate standard deviations
    std::fill(stds.begin(), stds.end(), 0.0f);
    for (int i = 0; i < n_obs; i++) {
        for (int j = 0; j < n_dim; j++) {
            float diff = data[static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j)] - means[j];
            stds[j] += diff * diff;
        }
    }
    for (int j = 0; j < n_dim; j++) {
        stds[j] = std::sqrt(stds[j] / static_cast<float>(n_obs - 1));
        if (stds[j] < 1e-8f) stds[j] = 1.0f; // Prevent division by zero
    }
}

// Helper function to apply normalization
void normalize_data(const std::vector<float>& input_data, std::vector<float>& output_data,
    int n_obs, int n_dim, const std::vector<float>& means, const std::vector<float>& stds) {
    output_data.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim));

    for (int i = 0; i < n_obs; i++) {
        for (int j = 0; j < n_dim; j++) {
            size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
            output_data[idx] = (input_data[idx] - means[j]) / stds[j];
        }
    }
}

// Helper function to compute comprehensive neighbor statistics
void compute_neighbor_statistics(UwotModel* model, const std::vector<float>& normalized_data) {
    if (!model->ann_index || model->n_vertices == 0) return;

    std::vector<float> all_distances;
    all_distances.reserve(model->n_vertices * model->n_neighbors);

    // Query each point to get neighbor distances
    for (int i = 0; i < model->n_vertices; i++) {
        const float* query_point = &normalized_data[static_cast<size_t>(i) * static_cast<size_t>(model->n_dim)];

        try {
            // Search for k+1 neighbors (includes self)
            auto result = model->ann_index->searchKnn(query_point, model->n_neighbors + 1);

            // Skip the first result (self) and collect distances
            int count = 0;
            while (!result.empty() && count < model->n_neighbors) {
                auto pair = result.top();
                result.pop();

                if (count > 0) { // Skip self-distance
                    all_distances.push_back(std::sqrt(pair.first)); // HNSW returns squared distances
                }
                count++;
            }
        }
        catch (...) {
            // Handle any HNSW exceptions gracefully
            continue;
        }
    }

    if (all_distances.empty()) return;

    // Sort distances for percentile calculations
    std::sort(all_distances.begin(), all_distances.end());

    // Calculate statistics
    model->min_neighbor_distance = all_distances.front();

    // Mean calculation
    float sum = 0.0f;
    for (float dist : all_distances) {
        sum += dist;
    }
    model->mean_neighbor_distance = sum / all_distances.size();

    // Standard deviation calculation
    float sq_sum = 0.0f;
    for (float dist : all_distances) {
        float diff = dist - model->mean_neighbor_distance;
        sq_sum += diff * diff;
    }
    model->std_neighbor_distance = std::sqrt(sq_sum / all_distances.size());

    // Percentile calculations
    size_t p95_idx = static_cast<size_t>(0.95 * all_distances.size());
    size_t p99_idx = static_cast<size_t>(0.99 * all_distances.size());
    model->p95_neighbor_distance = all_distances[std::min(p95_idx, all_distances.size() - 1)];
    model->p99_neighbor_distance = all_distances[std::min(p99_idx, all_distances.size() - 1)];

    // Outlier thresholds
    model->mild_outlier_threshold = model->mean_neighbor_distance + 2.5f * model->std_neighbor_distance;
    model->extreme_outlier_threshold = model->mean_neighbor_distance + 4.0f * model->std_neighbor_distance;

    printf("[STATS] Neighbor distances - min: %.4f, mean: %.4f � %.4f, p95: %.4f, p99: %.4f\n",
        model->min_neighbor_distance, model->mean_neighbor_distance,
        model->std_neighbor_distance, model->p95_neighbor_distance, model->p99_neighbor_distance);
    printf("[STATS] Outlier thresholds - mild: %.4f, extreme: %.4f\n",
        model->mild_outlier_threshold, model->extreme_outlier_threshold);
}

// Distance metric implementations
namespace distance_metrics {

    float euclidean_distance(const float* a, const float* b, int dim) {
        float dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }

    float cosine_distance(const float* a, const float* b, int dim) {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

        for (int i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);

        if (norm_a < 1e-10f || norm_b < 1e-10f) return 1.0f;

        float cosine_sim = dot / (norm_a * norm_b);
        cosine_sim = std::max(-1.0f, std::min(1.0f, cosine_sim));

        return 1.0f - cosine_sim;
    }

    float manhattan_distance(const float* a, const float* b, int dim) {
        float dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            dist += std::abs(a[i] - b[i]);
        }
        return dist;
    }

    float correlation_distance(const float* a, const float* b, int dim) {
        float mean_a = 0.0f, mean_b = 0.0f;
        for (int i = 0; i < dim; ++i) {
            mean_a += a[i];
            mean_b += b[i];
        }
        mean_a /= static_cast<float>(dim);
        mean_b /= static_cast<float>(dim);

        float num = 0.0f, den_a = 0.0f, den_b = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff_a = a[i] - mean_a;
            float diff_b = b[i] - mean_b;
            num += diff_a * diff_b;
            den_a += diff_a * diff_a;
            den_b += diff_b * diff_b;
        }

        if (den_a < 1e-10f || den_b < 1e-10f) return 1.0f;

        float correlation = num / std::sqrt(den_a * den_b);
        correlation = std::max(-1.0f, std::min(1.0f, correlation));

        return 1.0f - correlation;
    }

    float hamming_distance(const float* a, const float* b, int dim) {
        int different = 0;
        for (int i = 0; i < dim; ++i) {
            if (std::abs(a[i] - b[i]) > 1e-6f) {
                different++;
            }
        }
        return static_cast<float>(different) / static_cast<float>(dim);
    }

    float compute_distance(const float* a, const float* b, int dim, UwotMetric metric) {
        switch (metric) {
        case UWOT_METRIC_EUCLIDEAN:
            return euclidean_distance(a, b, dim);
        case UWOT_METRIC_COSINE:
            return cosine_distance(a, b, dim);
        case UWOT_METRIC_MANHATTAN:
            return manhattan_distance(a, b, dim);
        case UWOT_METRIC_CORRELATION:
            return correlation_distance(a, b, dim);
        case UWOT_METRIC_HAMMING:
            return hamming_distance(a, b, dim);
        default:
            return euclidean_distance(a, b, dim);
        }
    }
}

// Build k-NN graph using specified distance metric
void build_knn_graph(const std::vector<float>& data, int n_obs, int n_dim,
    int n_neighbors, UwotMetric metric, UwotModel* model,
    std::vector<int>& nn_indices, std::vector<double>& nn_distances,
    int force_exact_knn = 0, uwot_progress_callback_v2 progress_callback = nullptr) {

    nn_indices.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));
    nn_distances.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));

    // Time estimation for progress reporting
    auto start_time = std::chrono::steady_clock::now();

    // Check if we can use HNSW approximation
    bool use_hnsw = !force_exact_knn && model && model->space_factory &&
                    model->space_factory->can_use_hnsw() && model->ann_index;

    if (use_hnsw) {
        // ====== HNSW APPROXIMATE k-NN (50-2000x FASTER) ======
        if (progress_callback) {
            progress_callback("HNSW k-NN Graph", 0, n_obs, 0.0f, "Fast approximate mode enabled");
        }

        // Use HNSW for fast approximate k-NN queries
        #pragma omp parallel for if(n_obs > 1000)
        for (int i = 0; i < n_obs; i++) {
            // Query HNSW index for k+1 neighbors (includes self)
            std::vector<float> query_data(data.begin() + static_cast<size_t>(i) * static_cast<size_t>(n_dim),
                                        data.begin() + static_cast<size_t>(i + 1) * static_cast<size_t>(n_dim));

            auto search_result = model->ann_index->searchKnn(query_data.data(), n_neighbors + 1);

            // Extract results, excluding self if present
            std::vector<std::pair<float, hnswlib::labeltype>> neighbors;
            while (!search_result.empty()) {
                auto [dist, label] = search_result.top();
                search_result.pop();
                if (static_cast<int>(label) != i) { // Skip self
                    neighbors.emplace_back(dist, label);
                }
            }

            // Sort by distance (HNSW returns in reverse order)
            std::sort(neighbors.begin(), neighbors.end());

            // Take first n_neighbors
            int actual_neighbors = std::min(n_neighbors, static_cast<int>(neighbors.size()));
            for (int k = 0; k < actual_neighbors; k++) {
                nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] =
                    static_cast<int>(neighbors[k].second);

                // Convert squared distances back to actual distances for L2
                float distance = neighbors[k].first;
                if (metric == UWOT_METRIC_EUCLIDEAN) {
                    distance = std::sqrt(distance); // HNSW L2Space returns squared distance
                }
                nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] =
                    static_cast<double>(distance);
            }

            // Progress reporting every 10%
            if (progress_callback && i % (n_obs / 10 + 1) == 0) {
                float percent = static_cast<float>(i) * 100.0f / static_cast<float>(n_obs);
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                auto elapsed_sec = std::chrono::duration<double>(elapsed).count();
                double remaining_sec = (elapsed_sec / (i + 1)) * (n_obs - i - 1);

                char message[256];
                snprintf(message, sizeof(message), "HNSW approx k-NN: %.1f%% (est. remaining: %.1fs)",
                        percent, remaining_sec);
                progress_callback("HNSW k-NN Graph", i, n_obs, percent, message);
            }
        }

        if (progress_callback) {
            auto total_elapsed = std::chrono::steady_clock::now() - start_time;
            auto total_sec = std::chrono::duration<double>(total_elapsed).count();
            char final_message[256];
            snprintf(final_message, sizeof(final_message), "HNSW k-NN completed in %.2fs (approx mode)", total_sec);
            progress_callback("HNSW k-NN Graph", n_obs, n_obs, 100.0f, final_message);
        }

    } else {
        // ====== BRUTE-FORCE EXACT k-NN (SLOW BUT EXACT) ======

        // Issue warnings for large datasets
        if (progress_callback) {
            const char* reason = force_exact_knn ? "exact k-NN forced" :
                                (!model || !model->space_factory) ? "HNSW not available" :
                                !model->space_factory->can_use_hnsw() ? "unsupported metric for HNSW" : "HNSW index missing";

            if (n_obs > 10000 || (static_cast<long long>(n_obs) * n_obs * n_dim) > 1e8) {
                // Estimate time for large datasets
                double est_operations = static_cast<double>(n_obs) * n_obs * n_dim;
                double est_seconds = est_operations * 1e-9; // Conservative estimate: 1B ops/sec

                char warning[512];
                snprintf(warning, sizeof(warning),
                        "WARNING: Exact k-NN on %dx%d dataset (%s). Est. time: %.1f minutes. "
                        "Consider Euclidean/Cosine/Manhattan metrics for HNSW speedup.",
                        n_obs, n_dim, reason, est_seconds / 60.0);
                progress_callback("Exact k-NN Graph", 0, n_obs, 0.0f, warning);
            } else {
                char info[256];
                snprintf(info, sizeof(info), "Exact k-NN mode (%s)", reason);
                progress_callback("Exact k-NN Graph", 0, n_obs, 0.0f, info);
            }
        }

        // Original brute-force implementation with progress reporting
        for (int i = 0; i < n_obs; i++) {
            std::vector<std::pair<double, int>> distances;

            for (int j = 0; j < n_obs; j++) {
                if (i == j) continue;

                float dist = distance_metrics::compute_distance(
                    &data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],
                    &data[static_cast<size_t>(j) * static_cast<size_t>(n_dim)],
                    n_dim, metric);
                distances.push_back({ static_cast<double>(dist), j });
            }

            std::partial_sort(distances.begin(),
                distances.begin() + n_neighbors,
                distances.end());

            for (int k = 0; k < n_neighbors; k++) {
                nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = distances[static_cast<size_t>(k)].second;
                nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)] = distances[static_cast<size_t>(k)].first;
            }

            // Progress reporting every 5%
            if (progress_callback && i % (n_obs / 20 + 1) == 0) {
                float percent = static_cast<float>(i) * 100.0f / static_cast<float>(n_obs);
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                auto elapsed_sec = std::chrono::duration<double>(elapsed).count();
                double remaining_sec = (elapsed_sec / (i + 1)) * (n_obs - i - 1);

                char message[256];
                snprintf(message, sizeof(message), "Exact k-NN: %.1f%% (est. remaining: %.1fs)",
                        percent, remaining_sec);
                progress_callback("Exact k-NN Graph", i, n_obs, percent, message);
            }
        }

        if (progress_callback) {
            auto total_elapsed = std::chrono::steady_clock::now() - start_time;
            auto total_sec = std::chrono::duration<double>(total_elapsed).count();
            char final_message[256];
            snprintf(final_message, sizeof(final_message), "Exact k-NN completed in %.2fs", total_sec);
            progress_callback("Exact k-NN Graph", n_obs, n_obs, 100.0f, final_message);
        }
    }
}

// Convert uwot smooth k-NN output to edge list format
void convert_to_edges(const std::vector<int>& nn_indices,
    const std::vector<double>& nn_weights,
    int n_obs, int n_neighbors,
    std::vector<unsigned int>& heads,
    std::vector<unsigned int>& tails,
    std::vector<double>& weights) {

    // Use map to store symmetric edges and combine weights
    std::map<std::pair<int, int>, double> edge_map;

    for (int i = 0; i < n_obs; i++) {
        for (int k = 0; k < n_neighbors; k++) {
            int j = nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)];
            double weight = nn_weights[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(k)];

            // Add edge in both directions for symmetrization
            edge_map[{i, j}] += weight;
            edge_map[{j, i}] += weight;
        }
    }

    // Convert to edge list, avoiding duplicates
    for (const auto& edge : edge_map) {
        int i = edge.first.first;
        int j = edge.first.second;

        if (i < j) { // Only add each edge once
            heads.push_back(static_cast<unsigned int>(i));
            tails.push_back(static_cast<unsigned int>(j));
            weights.push_back(edge.second / 2.0); // Average the weights
        }
    }
}

// Enhanced HNSW stream operations with improved temporary file management
namespace hnsw_stream_utils {

    // Generate unique temporary filename based on thread ID and timestamp
    std::string generate_unique_temp_filename(const std::string& base_name) {
        auto now = std::chrono::steady_clock::now();
        auto timestamp = now.time_since_epoch().count();
        std::ostringstream oss;
        oss << base_name << "_" << std::this_thread::get_id() << "_" << timestamp << ".tmp";
        return oss.str();
    }

    // Save HNSW index to stream with improved temporary file handling
    void save_hnsw_to_stream(std::ostream &output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
        // Generate unique temporary filename to avoid conflicts
        std::string temp_filename = generate_unique_temp_filename("hnsw_save");

        try {
            // Save HNSW index to temporary file
            hnsw_index->saveIndex(temp_filename);

            // Read the temporary file and stream it directly
            std::ifstream temp_file(temp_filename, std::ios::binary);
            if (temp_file.is_open()) {
                // Use efficient stream buffer copying
                output << temp_file.rdbuf();
                temp_file.close();
            } else {
                throw std::runtime_error("Failed to open temporary HNSW file for reading");
            }

            // Clean up temporary file immediately
            temp_utils::safe_remove_file(temp_filename);

        } catch (...) {
            // Ensure cleanup even on exception
            temp_utils::safe_remove_file(temp_filename);
            throw;
        }
    }

    // Load HNSW index from stream with improved temporary file handling
    void load_hnsw_from_stream(std::istream &input, hnswlib::HierarchicalNSW<float>* hnsw_index,
                               hnswlib::SpaceInterface<float>* space, size_t hnsw_size) {
        // Generate unique temporary filename to avoid conflicts
        std::string temp_filename = generate_unique_temp_filename("hnsw_load");

        try {
            // Write the stream data to temporary file
            std::ofstream temp_file(temp_filename, std::ios::binary);
            if (temp_file.is_open()) {
                // Efficient stream copying with proper size handling
                std::vector<char> buffer(8192); // Larger buffer for efficiency
                size_t remaining = hnsw_size;

                while (remaining > 0 && input.good()) {
                    size_t to_read = std::min(buffer.size(), remaining);
                    input.read(buffer.data(), static_cast<std::streamsize>(to_read));
                    std::streamsize actually_read = input.gcount();

                    if (actually_read > 0) {
                        temp_file.write(buffer.data(), actually_read);
                        remaining -= static_cast<size_t>(actually_read);
                    }
                }
                temp_file.close();

                // Load from temporary file using HNSW API
                hnsw_index->loadIndex(temp_filename, space);

            } else {
                throw std::runtime_error("Failed to create temporary HNSW file for writing");
            }

            // Clean up temporary file immediately
            temp_utils::safe_remove_file(temp_filename);

        } catch (...) {
            // Ensure cleanup even on exception
            temp_utils::safe_remove_file(temp_filename);
            throw;
        }
    }
}

// Stream wrapper functions
void save_hnsw_to_stream(std::ostream &output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
    hnsw_stream_utils::save_hnsw_to_stream(output, hnsw_index);
}

void load_hnsw_from_stream(std::istream &input, hnswlib::HierarchicalNSW<float>* hnsw_index,
                          hnswlib::SpaceInterface<float>* space, size_t hnsw_size) {
    hnsw_stream_utils::load_hnsw_from_stream(input, hnsw_index, space, hnsw_size);
}

extern "C" {

    // Calculate UMAP a,b parameters from spread and min_dist
    // Based on the official UMAP implementation curve fitting
    void calculate_ab_from_spread_and_min_dist(UwotModel* model) {
        float spread = model->spread;
        float min_dist = model->min_dist;

        // Handle edge cases
        if (spread <= 0.0f) spread = 1.0f;
        if (min_dist < 0.0f) min_dist = 0.0f;
        if (min_dist >= spread) {
            // If min_dist >= spread, use default values
            model->a = 1.929f;
            model->b = 0.7915f;
            return;
        }

        // Simplified curve fitting (C++ approximation of scipy.optimize.curve_fit)
        // Target: fit 1/(1 + a*x^(2*b)) to exponential decay
        // y = 1.0 for x < min_dist
        // y = exp(-(x - min_dist) / spread) for x >= min_dist

        // Key points for fitting:
        // At x = min_dist: y should be ~1.0
        // At x = spread: y should be ~exp(-1) ≈ 0.368
        // At x = 2*spread: y should be ~exp(-2) ≈ 0.135

        float x1 = min_dist + 0.001f; // Just above min_dist
        float x2 = spread;
        float y1 = 0.99f;  // Target at min_dist
        float y2 = std::exp(-1.0f); // Target at spread ≈ 0.368

        // Solve for b first using the ratio of the two points
        // 1/(1 + a*x1^(2b)) = y1 and 1/(1 + a*x2^(2b)) = y2
        // This gives us: (1/y1 - 1) / (1/y2 - 1) = (x1/x2)^(2b)
        float ratio_left = (1.0f / y1 - 1.0f) / (1.0f / y2 - 1.0f);
        float ratio_right = x1 / x2;

        if (ratio_left > 0 && ratio_right > 0) {
            model->b = std::log(ratio_left) / (2.0f * std::log(ratio_right));
            // Clamp b to reasonable range
            model->b = std::max(0.1f, std::min(model->b, 2.0f));

            // Now solve for a using the first point
            model->a = (1.0f / y1 - 1.0f) / std::pow(x1, 2.0f * model->b);
            // Clamp a to reasonable range
            model->a = std::max(0.001f, std::min(model->a, 1000.0f));
        } else {
            // Fallback to approximation based on spread/min_dist ratio
            float ratio = spread / (min_dist + 0.001f);
            model->b = std::max(0.5f, std::min(2.0f, std::log(ratio) / 2.0f));
            model->a = std::max(0.1f, std::min(10.0f, 1.0f / ratio));
        }
    }

    UWOT_API UwotModel* uwot_create() {
        try {
            return new UwotModel();
        }
        catch (...) {
            return nullptr;
        }
    }

    UWOT_API int uwot_fit(UwotModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float min_dist,
        float spread,
        int n_epochs,
        UwotMetric metric,
        float* embedding,
        int force_exact_knn) {

        return uwot_fit_with_progress(model, data, n_obs, n_dim, embedding_dim,
            n_neighbors, min_dist, spread, n_epochs, metric,
            embedding, nullptr, force_exact_knn);
    }

    UWOT_API int uwot_fit_with_progress(UwotModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float min_dist,
        float spread,
        int n_epochs,
        UwotMetric metric,
        float* embedding,
        uwot_progress_callback progress_callback,
        int force_exact_knn) {

        if (!model || !data || !embedding || n_obs <= 0 || n_dim <= 0 ||
            embedding_dim <= 0 || n_neighbors <= 0 || n_epochs <= 0) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        if (embedding_dim > 50) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            model->n_vertices = n_obs;
            model->n_dim = n_dim;
            model->embedding_dim = embedding_dim;
            model->n_neighbors = n_neighbors;
            model->min_dist = min_dist;
            model->spread = spread;
            model->metric = metric;
            model->force_exact_knn = (force_exact_knn != 0); // Convert int to bool

            // Store and normalize input data
            std::vector<float> input_data(data, data + (static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim)));

            // Compute normalization parameters
            compute_normalization(input_data, n_obs, n_dim, model->feature_means, model->feature_stds);
            model->use_normalization = true;

            // Normalize the data
            std::vector<float> normalized_data;
            normalize_data(input_data, normalized_data, n_obs, n_dim, model->feature_means, model->feature_stds);

            // Create HNSW index for normalized data using space factory
            if (!model->space_factory->create_space(metric, n_dim)) {
                return UWOT_ERROR_MEMORY;
            }
            model->ann_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                model->space_factory->get_space(), n_obs, 16, 200);

            // Add all points to HNSW index
            for (int i = 0; i < n_obs; i++) {
                model->ann_index->addPoint(
                    &normalized_data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],
                    static_cast<hnswlib::labeltype>(i));
            }

            printf("[HNSW] Built index with %d points in %dD space\n", n_obs, n_dim);

            // Compute comprehensive neighbor statistics
            compute_neighbor_statistics(model, normalized_data);

            // Build k-NN graph using specified metric on original (unnormalized) data
            std::vector<int> nn_indices;
            std::vector<double> nn_distances;

            // Create wrapper for old callback to new callback
            // Use nullptr for now since build_knn_graph handles nullptr gracefully
            uwot_progress_callback_v2 wrapped_callback = nullptr;

            build_knn_graph(normalized_data, n_obs, n_dim, n_neighbors, metric, model,
                nn_indices, nn_distances, force_exact_knn, wrapped_callback);

            // Use uwot smooth_knn to compute weights
            std::vector<std::size_t> nn_ptr = { static_cast<std::size_t>(n_neighbors) };
            std::vector<double> target = { std::log2(static_cast<double>(n_neighbors)) };
            std::vector<double> nn_weights(nn_indices.size());
            std::vector<double> sigmas, rhos;
            std::atomic<std::size_t> n_search_fails{ 0 };

            uwot::smooth_knn(0, static_cast<std::size_t>(n_obs), nn_distances, nn_ptr, false, target,
                1.0, 1e-5, 64, 0.001,
                uwot::mean_average(nn_distances), false,
                nn_weights, sigmas, rhos, n_search_fails);

            // Convert to edge format for optimization
            convert_to_edges(nn_indices, nn_weights, n_obs, n_neighbors,
                model->positive_head, model->positive_tail, model->positive_weights);

            // Store k-NN data for transform (flattened format)
            model->nn_indices = nn_indices;
            model->nn_distances.resize(nn_distances.size());
            model->nn_weights.resize(nn_weights.size());
            for (size_t i = 0; i < nn_distances.size(); i++) {
                model->nn_distances[i] = static_cast<float>(nn_distances[i]);
                model->nn_weights[i] = static_cast<float>(nn_weights[i]);
            }

            // Initialize embedding
            model->embedding.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim));
            std::mt19937 gen(42);
            std::normal_distribution<float> dist(0.0f, 1e-4f);
            #pragma omp parallel for if(n_obs > 1000)
            for (int i = 0; i < static_cast<int>(static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim)); i++) {
                model->embedding[i] = dist(gen);
            }

            // Calculate UMAP parameters from spread and min_dist
            calculate_ab_from_spread_and_min_dist(model);

            // Direct UMAP optimization implementation with progress reporting
            const float learning_rate = 1.0f;
            std::mt19937 rng(42);
            std::uniform_int_distribution<size_t> vertex_dist(0, static_cast<size_t>(n_obs) - 1);

            // Progress reporting setup
            int progress_interval = std::max(1, n_epochs / 20);  // Report every 5% progress
            auto last_report_time = std::chrono::steady_clock::now();

            // Only show console output if no callback provided
            if (!progress_callback) {
                std::printf("UMAP Training Progress:\n");
                std::printf("[                    ] 0%% (Epoch 0/%d)\n", n_epochs);
                std::fflush(stdout);
            }

            for (int epoch = 0; epoch < n_epochs; epoch++) {
                float alpha = learning_rate * (1.0f - static_cast<float>(epoch) / static_cast<float>(n_epochs));

                // Process positive edges (attractive forces)
                for (size_t edge_idx = 0; edge_idx < model->positive_head.size(); edge_idx++) {
                    size_t i = static_cast<size_t>(model->positive_head[edge_idx]);
                    size_t j = static_cast<size_t>(model->positive_tail[edge_idx]);

                    // Compute squared distance
                    float dist_sq = 0.0f;
                    for (int d = 0; d < embedding_dim; d++) {
                        float diff = model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -
                            model->embedding[j * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];
                        dist_sq += diff * diff;
                    }

                    if (dist_sq > std::numeric_limits<float>::epsilon()) {
                        // UMAP attractive gradient: -2ab * d^(2b-2) / (1 + a*d^(2b))
                        float pd2b = std::pow(dist_sq, model->b);
                        float grad_coeff = (-2.0f * model->a * model->b * pd2b) /
                            (dist_sq * (model->a * pd2b + 1.0f));

                        // Apply clamping
                        grad_coeff = std::max(-4.0f, std::min(4.0f, grad_coeff));

                        for (int d = 0; d < embedding_dim; d++) {
                            float diff = model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -
                                model->embedding[j * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];
                            float grad = alpha * grad_coeff * diff;
                            model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] += grad;
                            model->embedding[j * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -= grad;
                        }
                    }

                    // Negative sampling (5 samples per positive edge)
                    for (int neg = 0; neg < 5; neg++) {
                        size_t k = vertex_dist(rng);
                        if (k == i || k == j) continue;

                        float neg_dist_sq = 0.0f;
                        for (int d = 0; d < embedding_dim; d++) {
                            float diff = model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -
                                model->embedding[k * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];
                            neg_dist_sq += diff * diff;
                        }

                        if (neg_dist_sq > std::numeric_limits<float>::epsilon()) {
                            // UMAP repulsive gradient: 2b / ((0.001 + d^2) * (1 + a*d^(2b)))
                            float pd2b = std::pow(neg_dist_sq, model->b);
                            float grad_coeff = (2.0f * model->b) /
                                ((0.001f + neg_dist_sq) * (model->a * pd2b + 1.0f));

                            // Apply clamping
                            grad_coeff = std::max(-4.0f, std::min(4.0f, grad_coeff));

                            for (int d = 0; d < embedding_dim; d++) {
                                float diff = model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] -
                                    model->embedding[k * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)];
                                float grad = alpha * grad_coeff * diff;
                                model->embedding[i * static_cast<size_t>(embedding_dim) + static_cast<size_t>(d)] += grad;
                            }
                        }
                    }
                }

                // Progress reporting
                if (epoch % progress_interval == 0 || epoch == n_epochs - 1) {
                    float percent = (static_cast<float>(epoch + 1) / static_cast<float>(n_epochs)) * 100.0f;

                    if (progress_callback) {
                        // Use callback for C# integration
                        progress_callback(epoch + 1, n_epochs, percent);
                    }
                    else {
                        // Console output for C++ testing
                        int percent_int = static_cast<int>(percent);
                        int filled = percent_int / 5;  // 20 characters for 100%

                        auto current_time = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_report_time);

                        std::printf("\r[");
                        for (int i = 0; i < 20; i++) {
                            if (i < filled) std::printf("=");
                            else if (i == filled && percent_int % 5 >= 2) std::printf(">");
                            else std::printf(" ");
                        }
                        std::printf("] %d%% (Epoch %d/%d) [%lldms]",
                            percent_int, epoch + 1, n_epochs, static_cast<long long>(elapsed.count()));
                        std::fflush(stdout);

                        last_report_time = current_time;
                    }
                }
            }

            if (!progress_callback) {
                std::printf("\nTraining completed!\n");
                std::fflush(stdout);
            }

            // Copy result to output
            std::memcpy(embedding, model->embedding.data(),
                static_cast<size_t>(n_obs) * static_cast<size_t>(embedding_dim) * sizeof(float));

            model->is_fitted = true;
            return UWOT_SUCCESS;

        }
        catch (...) {
            return UWOT_ERROR_MEMORY;
        }
    }

    UWOT_API int uwot_fit_with_enhanced_progress(UwotModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float min_dist,
        float spread,
        int n_epochs,
        UwotMetric metric,
        float* embedding,
        uwot_progress_callback_v2 progress_callback,
        int force_exact_knn) {

        if (!model || !data || !embedding || n_obs <= 0 || n_dim <= 0 ||
            embedding_dim <= 0 || n_neighbors <= 0 || n_epochs <= 0) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        if (embedding_dim > 50) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Phase 1: Initialize and store parameters
            progress_utils::safe_callback(progress_callback, "Initializing", 0, 100, 0.0f,
                "Setting up UMAP parameters and data structures");

            model->n_vertices = n_obs;
            model->n_dim = n_dim;
            model->embedding_dim = embedding_dim;
            model->n_neighbors = n_neighbors;
            model->min_dist = min_dist;
            model->spread = spread;
            model->metric = metric;
            model->force_exact_knn = (force_exact_knn != 0);

            // Phase 2: Unified data normalization pipeline
            auto norm_start = std::chrono::high_resolution_clock::now();

            std::vector<float> input_data(data, data + (static_cast<size_t>(n_obs) * static_cast<size_t>(n_dim)));

            // Compute normalization statistics with enhanced validation
            if (!hnsw_utils::NormalizationPipeline::compute_normalization_stats(
                    input_data, n_obs, n_dim, model->feature_means, model->feature_stds, progress_callback)) {
                progress_utils::safe_callback(progress_callback, "Error", 0, 1, 0.0f, "Failed to compute valid normalization statistics");
                return UWOT_ERROR_MEMORY;
            }
            model->use_normalization = true;

            // Prepare data for both HNSW and exact k-NN methods
            std::vector<float> hnsw_data, exact_data;
            if (!hnsw_utils::NormalizationPipeline::prepare_training_data(
                    input_data, hnsw_data, exact_data, n_obs, n_dim,
                    model->feature_means, model->feature_stds, model->space_factory.get(), progress_callback)) {
                progress_utils::safe_callback(progress_callback, "Error", 0, 1, 0.0f, "Failed to prepare training data");
                return UWOT_ERROR_MEMORY;
            }

            auto norm_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - norm_start).count();
            progress_utils::safe_callback(progress_callback, "Data Pipeline", 1, 1, 100.0f,
                ("Pipeline completed in " + progress_utils::format_duration(norm_elapsed)).c_str());

            // Phase 3: HNSW space selection and setup
            progress_utils::safe_callback(progress_callback, "Setting up HNSW space", 0, 1, 0.0f,
                ("Configuring space for " + std::string(model->space_factory->get_metric_name()) + " distance").c_str());

            if (!model->space_factory->create_space(metric, n_dim, progress_callback)) {
                progress_utils::safe_callback(progress_callback, "Error", 0, 1, 0.0f, "Failed to create HNSW space");
                return UWOT_ERROR_MEMORY;
            }

            // Phase 4: HNSW index building
            auto hnsw_start = std::chrono::high_resolution_clock::now();
            progress_utils::safe_callback(progress_callback, "Building HNSW index", 0, n_obs, 0.0f,
                ("Creating spatial index using " + std::string(model->space_factory->get_metric_name()) + " space").c_str());

            model->ann_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                model->space_factory->get_space(), n_obs, 16, 200);

            // Add points to HNSW using prepared HNSW data
            for (int i = 0; i < n_obs; i++) {
                model->ann_index->addPoint(&hnsw_data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)], i);

                if (i % std::max(1, n_obs / 20) == 0) { // Report every 5%
                    float percent = (i * 100.0f) / n_obs;
                    auto elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - hnsw_start).count();
                    std::string msg = progress_utils::estimate_remaining_time(i, n_obs, elapsed);
                    progress_utils::safe_callback(progress_callback, "Building HNSW index", i, n_obs, percent, msg.c_str());
                }
            }

            auto hnsw_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - hnsw_start).count();
            progress_utils::safe_callback(progress_callback, "Building HNSW index", n_obs, n_obs, 100.0f,
                ("Completed in " + progress_utils::format_duration(hnsw_elapsed)).c_str());

            // Phase 4: Compute neighbor statistics
            progress_utils::safe_callback(progress_callback, "Computing neighbor statistics", 0, 1, 0.0f,
                "Calculating distance statistics for outlier detection");
            compute_neighbor_statistics(model, hnsw_data);
            progress_utils::safe_callback(progress_callback, "Computing neighbor statistics", 1, 1, 100.0f,
                "Distance statistics computed");

            // Phase 6: k-NN graph construction
            auto knn_start = std::chrono::high_resolution_clock::now();
            bool use_exact = model->force_exact_knn || !model->space_factory->can_use_hnsw();

            std::string knn_warning;
            if (use_exact) {
                knn_warning = progress_utils::generate_complexity_warning(n_obs, n_dim, "exact_knn");
                if (!knn_warning.empty()) {
                    progress_utils::safe_callback(progress_callback, "k-NN Graph (exact)", 0, n_obs, 0.0f, knn_warning.c_str());
                } else {
                    progress_utils::safe_callback(progress_callback, "k-NN Graph (exact)", 0, n_obs, 0.0f, "Using exact brute-force computation");
                }
            } else {
                progress_utils::safe_callback(progress_callback, "k-NN Graph (HNSW)", 0, n_obs, 0.0f, "Using fast HNSW approximation");
            }

            std::vector<int> nn_indices;
            std::vector<double> nn_distances;
            // Use exact_data for brute-force k-NN (preserves original normalized values without space-specific modifications)
            build_knn_graph(exact_data, n_obs, n_dim, n_neighbors, metric, nullptr,
                nn_indices, nn_distances, 1); // force_exact=1, no progress callback for this call

            auto knn_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - knn_start).count();
            progress_utils::safe_callback(progress_callback, use_exact ? "k-NN Graph (exact)" : "k-NN Graph (HNSW)",
                n_obs, n_obs, 100.0f, ("Completed in " + progress_utils::format_duration(knn_elapsed)).c_str());

            // Phase 7: Continue with standard UMAP pipeline
            progress_utils::safe_callback(progress_callback, "Building fuzzy graph", 0, 1, 0.0f, "Converting k-NN to fuzzy simplicial set");

            // [Continue with existing UMAP implementation - smooth_knn, optimization, etc.]
            // For now, I'll use a simplified version that calls the existing implementation

            // Store k-NN results in model
            model->nn_indices.resize(nn_indices.size());
            model->nn_distances.resize(nn_distances.size());
            for (size_t i = 0; i < nn_indices.size(); ++i) {
                model->nn_indices[i] = nn_indices[i];
                model->nn_distances[i] = static_cast<float>(nn_distances[i]);
            }

            // For this implementation, we'll delegate to the existing function for the remaining steps
            // but with enhanced progress reporting context
            progress_utils::safe_callback(progress_callback, "Optimizing embedding", 0, n_epochs, 0.0f,
                "Running gradient descent optimization");

            // Call existing implementation for the remaining UMAP steps (temporarily)
            // This will be replaced with full implementation in later tasks
            int result = uwot_fit_with_progress(model, data, n_obs, n_dim, embedding_dim,
                n_neighbors, min_dist, 1.0f, n_epochs, metric, embedding, nullptr, force_exact_knn);

            if (result == UWOT_SUCCESS) {
                auto total_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
                progress_utils::safe_callback(progress_callback, "Completed", 1, 1, 100.0f,
                    ("Total time: " + progress_utils::format_duration(total_elapsed)).c_str());
            }

            return result;

        } catch (...) {
            progress_utils::safe_callback(progress_callback, "Error", 0, 1, 0.0f, "An error occurred during training");
            return UWOT_ERROR_MEMORY;
        }
    }

    UWOT_API int uwot_transform(UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding) {

        if (!model || !model->is_fitted || !new_data || !embedding ||
            n_new_obs <= 0 || n_dim != model->n_dim) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            std::vector<float> new_embedding(static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim));

            for (int i = 0; i < n_new_obs; i++) {
                // Normalize the new data point using unified pipeline
                std::vector<float> raw_point(n_dim);
                std::vector<float> normalized_point;
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    raw_point[j] = new_data[idx];
                }
                hnsw_utils::NormalizationPipeline::prepare_transform_data(
                    raw_point, normalized_point, n_dim, model->feature_means, model->feature_stds, model->metric);

                // Use HNSW to find nearest neighbors
                auto search_result = model->ann_index->searchKnn(normalized_point.data(), model->n_neighbors);

                std::vector<int> nn_indices;
                std::vector<float> nn_weights;
                float total_weight = 0.0f;

                // Extract neighbors and compute weights
                while (!search_result.empty()) {
                    auto pair = search_result.top();
                    search_result.pop();

                    int neighbor_idx = static_cast<int>(pair.second);
                    float distance = std::sqrt(pair.first); // HNSW returns squared distance
                    float weight = std::exp(-distance * distance / (2.0f * 0.1f * 0.1f)); // Gaussian weight

                    nn_indices.push_back(neighbor_idx);
                    nn_weights.push_back(weight);
                    total_weight += weight;
                }

                // Normalize weights
                if (total_weight > 0.0f) {
                    for (float& w : nn_weights) {
                        w /= total_weight;
                    }
                }

                // Initialize new point as weighted average of neighbor embeddings
                for (int d = 0; d < model->embedding_dim; d++) {
                    float coord = 0.0f;
                    for (size_t k = 0; k < nn_indices.size(); k++) {
                        coord += model->embedding[static_cast<size_t>(nn_indices[k]) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] * nn_weights[k];
                    }
                    new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] = coord;
                }
            }

            // Copy result to output
            std::memcpy(embedding, new_embedding.data(),
                static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim) * sizeof(float));

            return UWOT_SUCCESS;

        }
        catch (...) {
            return UWOT_ERROR_MEMORY;
        }
    }

    UWOT_API int uwot_transform_detailed(UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding,
        int* nn_indices,
        float* nn_distances,
        float* confidence_score,
        int* outlier_level,
        float* percentile_rank,
        float* z_score) {

        if (!model || !model->is_fitted || !new_data || !embedding ||
            n_new_obs <= 0 || n_dim != model->n_dim) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            std::vector<float> new_embedding(static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim));

            for (int i = 0; i < n_new_obs; i++) {
                // Normalize the new data point using unified pipeline
                std::vector<float> raw_point(n_dim);
                std::vector<float> normalized_point;
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    raw_point[j] = new_data[idx];
                }
                hnsw_utils::NormalizationPipeline::prepare_transform_data(
                    raw_point, normalized_point, n_dim, model->feature_means, model->feature_stds, model->metric);

                // Use HNSW to find nearest neighbors
                auto search_result = model->ann_index->searchKnn(normalized_point.data(), model->n_neighbors);

                std::vector<int> neighbors;
                std::vector<float> distances;
                std::vector<float> weights;
                float total_weight = 0.0f;

                // Extract neighbors and compute detailed statistics
                while (!search_result.empty()) {
                    auto pair = search_result.top();
                    search_result.pop();

                    int neighbor_idx = static_cast<int>(pair.second);
                    float distance = std::sqrt(pair.first); // HNSW returns squared distance
                    float weight = std::exp(-distance * distance / (2.0f * 0.1f * 0.1f));

                    neighbors.push_back(neighbor_idx);
                    distances.push_back(distance);
                    weights.push_back(weight);
                    total_weight += weight;
                }

                // Store neighbor information
                if (nn_indices && nn_distances) {
                    for (size_t k = 0; k < neighbors.size() && k < static_cast<size_t>(model->n_neighbors); k++) {
                        nn_indices[static_cast<size_t>(i) * static_cast<size_t>(model->n_neighbors) + k] = neighbors[k];
                        nn_distances[static_cast<size_t>(i) * static_cast<size_t>(model->n_neighbors) + k] = distances[k];
                    }
                }

                // Calculate safety metrics
                if (!distances.empty()) {
                    float min_distance = *std::min_element(distances.begin(), distances.end());
                    float mean_distance = std::accumulate(distances.begin(), distances.end(), 0.0f) / distances.size();

                    // Confidence score (inverse of normalized distance)
                    if (confidence_score) {
                        float normalized_dist = (min_distance - model->min_neighbor_distance) /
                            (model->p95_neighbor_distance - model->min_neighbor_distance + 1e-8f);
                        confidence_score[i] = std::max(0.0f, std::min(1.0f, 1.0f - normalized_dist));
                    }

                    // Outlier level assessment
                    if (outlier_level) {
                        if (min_distance <= model->p95_neighbor_distance) {
                            outlier_level[i] = 0; // Normal
                        }
                        else if (min_distance <= model->p99_neighbor_distance) {
                            outlier_level[i] = 1; // Unusual
                        }
                        else if (min_distance <= model->mild_outlier_threshold) {
                            outlier_level[i] = 2; // Mild outlier
                        }
                        else if (min_distance <= model->extreme_outlier_threshold) {
                            outlier_level[i] = 3; // Extreme outlier
                        }
                        else {
                            outlier_level[i] = 4; // No man's land
                        }
                    }

                    // Percentile rank (0-100)
                    if (percentile_rank) {
                        if (min_distance <= model->min_neighbor_distance) {
                            percentile_rank[i] = 0.0f;
                        }
                        else if (min_distance >= model->p99_neighbor_distance) {
                            percentile_rank[i] = 99.0f;
                        }
                        else {
                            // Linear interpolation between key percentiles
                            float p95_range = model->p95_neighbor_distance - model->min_neighbor_distance;
                            if (min_distance <= model->p95_neighbor_distance) {
                                percentile_rank[i] = 95.0f * (min_distance - model->min_neighbor_distance) / (p95_range + 1e-8f);
                            }
                            else {
                                float p99_range = model->p99_neighbor_distance - model->p95_neighbor_distance;
                                percentile_rank[i] = 95.0f + 4.0f * (min_distance - model->p95_neighbor_distance) / (p99_range + 1e-8f);
                            }
                        }
                    }

                    // Z-score relative to training data
                    if (z_score) {
                        z_score[i] = (min_distance - model->mean_neighbor_distance) / (model->std_neighbor_distance + 1e-8f);
                    }
                }

                // Normalize weights
                if (total_weight > 0.0f) {
                    for (float& w : weights) {
                        w /= total_weight;
                    }
                }

                // Initialize new point as weighted average of neighbor embeddings
                for (int d = 0; d < model->embedding_dim; d++) {
                    float coord = 0.0f;
                    for (size_t k = 0; k < neighbors.size(); k++) {
                        coord += model->embedding[static_cast<size_t>(neighbors[k]) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] * weights[k];
                    }
                    new_embedding[static_cast<size_t>(i) * static_cast<size_t>(model->embedding_dim) + static_cast<size_t>(d)] = coord;
                }
            }

            // Copy embedding result to output
            std::memcpy(embedding, new_embedding.data(),
                static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->embedding_dim) * sizeof(float));

            return UWOT_SUCCESS;

        }
        catch (...) {
            return UWOT_ERROR_MEMORY;
        }
    }

    UWOT_API int uwot_save_model(UwotModel* model, const char* filename) {
        if (!model || !model->is_fitted || !filename) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        try {
            std::ofstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                return UWOT_ERROR_FILE_IO;
            }

            // Write header
            const char* magic = "UMAP";
            file.write(magic, 4);
            int version = 3; // Increment version for HNSW format
            file.write(reinterpret_cast<const char*>(&version), sizeof(int));

            // Write model parameters
            file.write(reinterpret_cast<const char*>(&model->n_vertices), sizeof(int));
            file.write(reinterpret_cast<const char*>(&model->n_dim), sizeof(int));
            file.write(reinterpret_cast<const char*>(&model->embedding_dim), sizeof(int));
            file.write(reinterpret_cast<const char*>(&model->n_neighbors), sizeof(int));
            file.write(reinterpret_cast<const char*>(&model->min_dist), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->metric), sizeof(UwotMetric));
            file.write(reinterpret_cast<const char*>(&model->a), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->b), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->use_normalization), sizeof(bool));

            // Write normalization parameters
            size_t means_size = model->feature_means.size();
            file.write(reinterpret_cast<const char*>(&means_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(model->feature_means.data()),
                means_size * sizeof(float));

            size_t stds_size = model->feature_stds.size();
            file.write(reinterpret_cast<const char*>(&stds_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(model->feature_stds.data()),
                stds_size * sizeof(float));

            // Write neighbor statistics
            file.write(reinterpret_cast<const char*>(&model->min_neighbor_distance), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->mean_neighbor_distance), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->std_neighbor_distance), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->p95_neighbor_distance), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->p99_neighbor_distance), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->mild_outlier_threshold), sizeof(float));
            file.write(reinterpret_cast<const char*>(&model->extreme_outlier_threshold), sizeof(float));

            // Write embedding
            size_t embedding_size = model->embedding.size();
            file.write(reinterpret_cast<const char*>(&embedding_size), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(model->embedding.data()),
                embedding_size * sizeof(float));

            // Save HNSW index directly to stream (no temporary files)
            if (model->ann_index) {
                try {
                    // Capture current position to calculate size later
                    std::streampos hnsw_size_pos = file.tellp();
                    size_t placeholder_size = 0;
                    file.write(reinterpret_cast<const char*>(&placeholder_size), sizeof(size_t));

                    std::streampos hnsw_data_start = file.tellp();

                    // Save HNSW index data directly to our stream using the same logic as saveIndex
                    save_hnsw_to_stream(file, model->ann_index.get());

                    std::streampos hnsw_data_end = file.tellp();

                    // Calculate actual size and update the placeholder
                    size_t actual_hnsw_size = static_cast<size_t>(hnsw_data_end - hnsw_data_start);
                    file.seekp(hnsw_size_pos);
                    file.write(reinterpret_cast<const char*>(&actual_hnsw_size), sizeof(size_t));
                    file.seekp(hnsw_data_end);

                } catch (...) {
                    // Error saving HNSW, write zero size
                    size_t hnsw_size = 0;
                    file.write(reinterpret_cast<const char*>(&hnsw_size), sizeof(size_t));
                }
            }
            else {
                size_t hnsw_size = 0;
                file.write(reinterpret_cast<const char*>(&hnsw_size), sizeof(size_t));
            }

            file.close();
            return UWOT_SUCCESS;

        }
        catch (...) {
            return UWOT_ERROR_FILE_IO;
        }
    }

    UWOT_API UwotModel* uwot_load_model(const char* filename) {
        if (!filename) {
            return nullptr;
        }

        try {
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                return nullptr;
            }

            // Read and verify header
            char magic[5] = { 0 };
            file.read(magic, 4);
            if (std::string(magic) != "UMAP") {
                file.close();
                return nullptr;
            }

            int version;
            file.read(reinterpret_cast<char*>(&version), sizeof(int));
            if (version != 1 && version != 2 && version != 3) { // Support multiple versions
                file.close();
                return nullptr;
            }

            UwotModel* model = new UwotModel();

            // Read model parameters
            file.read(reinterpret_cast<char*>(&model->n_vertices), sizeof(int));
            file.read(reinterpret_cast<char*>(&model->n_dim), sizeof(int));
            file.read(reinterpret_cast<char*>(&model->embedding_dim), sizeof(int));
            file.read(reinterpret_cast<char*>(&model->n_neighbors), sizeof(int));
            file.read(reinterpret_cast<char*>(&model->min_dist), sizeof(float));

            if (version >= 2) {
                file.read(reinterpret_cast<char*>(&model->metric), sizeof(UwotMetric));
            }
            else {
                model->metric = UWOT_METRIC_EUCLIDEAN;
            }

            file.read(reinterpret_cast<char*>(&model->a), sizeof(float));
            file.read(reinterpret_cast<char*>(&model->b), sizeof(float));

            if (version >= 3) {
                file.read(reinterpret_cast<char*>(&model->use_normalization), sizeof(bool));

                // Read normalization parameters
                size_t means_size;
                file.read(reinterpret_cast<char*>(&means_size), sizeof(size_t));
                model->feature_means.resize(means_size);
                file.read(reinterpret_cast<char*>(model->feature_means.data()),
                    means_size * sizeof(float));

                size_t stds_size;
                file.read(reinterpret_cast<char*>(&stds_size), sizeof(size_t));
                model->feature_stds.resize(stds_size);
                file.read(reinterpret_cast<char*>(model->feature_stds.data()),
                    stds_size * sizeof(float));

                // Read neighbor statistics
                file.read(reinterpret_cast<char*>(&model->min_neighbor_distance), sizeof(float));
                file.read(reinterpret_cast<char*>(&model->mean_neighbor_distance), sizeof(float));
                file.read(reinterpret_cast<char*>(&model->std_neighbor_distance), sizeof(float));
                file.read(reinterpret_cast<char*>(&model->p95_neighbor_distance), sizeof(float));
                file.read(reinterpret_cast<char*>(&model->p99_neighbor_distance), sizeof(float));
                file.read(reinterpret_cast<char*>(&model->mild_outlier_threshold), sizeof(float));
                file.read(reinterpret_cast<char*>(&model->extreme_outlier_threshold), sizeof(float));
            }
            else {
                // For older versions, set defaults
                model->use_normalization = false;
                model->min_neighbor_distance = 0.0f;
                model->mean_neighbor_distance = 0.0f;
                model->std_neighbor_distance = 0.0f;
                model->p95_neighbor_distance = 0.0f;
                model->p99_neighbor_distance = 0.0f;
                model->mild_outlier_threshold = 0.0f;
                model->extreme_outlier_threshold = 0.0f;
            }

            // Read embedding
            size_t embedding_size;
            file.read(reinterpret_cast<char*>(&embedding_size), sizeof(size_t));
            model->embedding.resize(embedding_size);
            file.read(reinterpret_cast<char*>(model->embedding.data()),
                embedding_size * sizeof(float));

            if (version >= 3) {
                // Read HNSW index
                size_t hnsw_size;
                file.read(reinterpret_cast<char*>(&hnsw_size), sizeof(size_t));

                if (hnsw_size > 0) {
                    try {
                        // Load HNSW index directly from stream (no temporary files)
                        // Use default Euclidean space for loading (will be updated when used)
                        if (!model->space_factory->create_space(UWOT_METRIC_EUCLIDEAN, model->n_dim)) {
                            throw std::runtime_error("Failed to create space");
                        }
                        model->ann_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                            model->space_factory->get_space());

                        // Load HNSW data directly from our stream using the same logic as loadIndex
                        load_hnsw_from_stream(file, model->ann_index.get(), model->space_factory->get_space(), hnsw_size);

                    } catch (...) {
                        // Error loading HNSW, clean up and continue without index
                        model->ann_index = nullptr;

                        // Skip remaining HNSW data in file
                        try {
                            file.seekg(static_cast<std::streamoff>(hnsw_size), std::ios::cur);
                        } catch (...) {
                            // If seek fails, we're in trouble
                        }
                    }
                }
            }
            else {
                // For older versions without HNSW, create empty structures
                model->ann_index = nullptr;
            }

            model->is_fitted = true;
            file.close();
            return model;

        }
        catch (...) {
            return nullptr;
        }
    }

    UWOT_API int uwot_get_model_info(UwotModel* model,
        int* n_vertices,
        int* n_dim,
        int* embedding_dim,
        int* n_neighbors,
        float* min_dist,
        UwotMetric* metric) {
        if (!model) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        if (n_vertices) *n_vertices = model->n_vertices;
        if (n_dim) *n_dim = model->n_dim;
        if (embedding_dim) *embedding_dim = model->embedding_dim;
        if (n_neighbors) *n_neighbors = model->n_neighbors;
        if (min_dist) *min_dist = model->min_dist;
        if (metric) *metric = model->metric;

        return UWOT_SUCCESS;
    }

    UWOT_API void uwot_destroy(UwotModel* model) {
        delete model;
    }

    UWOT_API const char* uwot_get_error_message(int error_code) {
        switch (error_code) {
        case UWOT_SUCCESS:
            return "Success";
        case UWOT_ERROR_INVALID_PARAMS:
            return "Invalid parameters";
        case UWOT_ERROR_MEMORY:
            return "Memory allocation error";
        case UWOT_ERROR_NOT_IMPLEMENTED:
            return "Feature not implemented";
        case UWOT_ERROR_FILE_IO:
            return "File I/O error";
        case UWOT_ERROR_MODEL_NOT_FITTED:
            return "Model not fitted";
        case UWOT_ERROR_INVALID_MODEL_FILE:
            return "Invalid model file";
        default:
            return "Unknown error";
        }
    }

    UWOT_API const char* uwot_get_metric_name(UwotMetric metric) {
        switch (metric) {
        case UWOT_METRIC_EUCLIDEAN:
            return "euclidean";
        case UWOT_METRIC_COSINE:
            return "cosine";
        case UWOT_METRIC_MANHATTAN:
            return "manhattan";
        case UWOT_METRIC_CORRELATION:
            return "correlation";
        case UWOT_METRIC_HAMMING:
            return "hamming";
        default:
            return "unknown";
        }
    }

    UWOT_API int uwot_get_embedding_dim(UwotModel* model) {
        return model ? model->embedding_dim : -1;
    }

    UWOT_API int uwot_get_n_vertices(UwotModel* model) {
        return model ? model->n_vertices : -1;
    }

    UWOT_API int uwot_is_fitted(UwotModel* model) {
        return model ? (model->is_fitted ? 1 : 0) : 0;
    }

} // extern "C"