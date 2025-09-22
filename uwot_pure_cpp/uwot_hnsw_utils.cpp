#include "uwot_hnsw_utils.h"
#include "uwot_progress_utils.h"
#include "lz4.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
#include <filesystem>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

// HNSW space factory implementation
namespace hnsw_utils {

    bool SpaceFactory::create_space(UwotMetric metric, int n_dim) {
        current_metric = metric;
        current_dim = n_dim;

        // Clean up existing spaces
        l2_space.reset();
        ip_space.reset();
        l1_space.reset();

        try {
            switch (metric) {
            case UWOT_METRIC_EUCLIDEAN:
                l2_space = std::make_unique<hnswlib::L2Space>(n_dim);
                return true;

            case UWOT_METRIC_COSINE:
                ip_space = std::make_unique<hnswlib::InnerProductSpace>(n_dim);
                return true;

            case UWOT_METRIC_MANHATTAN:
                l1_space = std::make_unique<L1Space>(n_dim);
                return true;

            default:
                return false; // Unsupported metric for HNSW
            }
        }
        catch (...) {
            return false;
        }
    }

    hnswlib::SpaceInterface<float>* SpaceFactory::get_space() {
        switch (current_metric) {
        case UWOT_METRIC_EUCLIDEAN:
            return l2_space.get();
        case UWOT_METRIC_COSINE:
            return ip_space.get();
        case UWOT_METRIC_MANHATTAN:
            return l1_space.get();
        default:
            return nullptr;
        }
    }

    bool SpaceFactory::can_use_hnsw() const {
        return current_metric == UWOT_METRIC_EUCLIDEAN ||
               current_metric == UWOT_METRIC_COSINE ||
               current_metric == UWOT_METRIC_MANHATTAN;
    }

    // HNSW stream utilities implementation
    namespace hnsw_stream_utils {

        std::string generate_unique_temp_filename(const std::string& prefix) {
            auto now = std::chrono::high_resolution_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1000, 9999);
            int random_suffix = dis(gen);

            std::ostringstream oss;
            oss << prefix << "_" << timestamp << "_" << random_suffix << ".tmp";
            return oss.str();
        }

        void save_hnsw_to_stream(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
            std::string temp_filename = generate_unique_temp_filename("hnsw_save");

            try {
                // Save HNSW index to temporary file
                hnsw_index->saveIndex(temp_filename);

                // Read the temporary file and stream it directly
                std::ifstream temp_file(temp_filename, std::ios::binary);
                if (!temp_file.is_open()) {
                    throw std::runtime_error("Failed to open temporary HNSW file for reading");
                }

                // Stream the file contents
                output << temp_file.rdbuf();
                temp_file.close();

                // Clean up temporary file
                temp_utils::safe_remove_file(temp_filename);
            }
            catch (...) {
                // Ensure cleanup on error
                temp_utils::safe_remove_file(temp_filename);
                throw;
            }
        }

        void load_hnsw_from_stream(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
            hnswlib::SpaceInterface<float>* space, size_t hnsw_size) {
            std::string temp_filename = generate_unique_temp_filename("hnsw_load");

            try {
                // Write stream data to temporary file
                std::ofstream temp_file(temp_filename, std::ios::binary);
                if (!temp_file.is_open()) {
                    throw std::runtime_error("Failed to create temporary HNSW file");
                }

                // Copy specified amount of data from stream to file
                std::vector<char> buffer(8192);
                size_t remaining = hnsw_size;
                while (remaining > 0 && input.good()) {
                    size_t to_read = std::min(remaining, buffer.size());
                    input.read(buffer.data(), to_read);
                    size_t actually_read = input.gcount();
                    if (actually_read > 0) {
                        temp_file.write(buffer.data(), actually_read);
                        remaining -= actually_read;
                    }
                    else {
                        break;
                    }
                }
                temp_file.close();

                // Load from temporary file
                hnsw_index->loadIndex(temp_filename, space);

                // Clean up temporary file
                temp_utils::safe_remove_file(temp_filename);
            }
            catch (...) {
                // Ensure cleanup on error
                temp_utils::safe_remove_file(temp_filename);
                throw;
            }
        }
    }

    // HNSW compression utilities
    void save_hnsw_to_stream_compressed(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
        std::string temp_filename = hnsw_stream_utils::generate_unique_temp_filename("hnsw_compressed");

        try {
            // Save HNSW index to temporary file
            hnsw_index->saveIndex(temp_filename);

            // Read the temporary file
            std::ifstream temp_file(temp_filename, std::ios::binary | std::ios::ate);
            if (!temp_file.is_open()) {
                throw std::runtime_error("Failed to open temporary HNSW file for compression");
            }

            std::streamsize file_size = temp_file.tellg();
            temp_file.seekg(0, std::ios::beg);

            std::vector<char> uncompressed_data(file_size);
            if (!temp_file.read(uncompressed_data.data(), file_size)) {
                throw std::runtime_error("Failed to read HNSW temporary file");
            }
            temp_file.close();

            // Compress with LZ4
            int max_compressed_size = LZ4_compressBound(static_cast<int>(file_size));
            std::vector<char> compressed_data(max_compressed_size);

            int compressed_size = LZ4_compress_default(
                uncompressed_data.data(), compressed_data.data(),
                static_cast<int>(file_size), max_compressed_size);

            if (compressed_size <= 0) {
                throw std::runtime_error("LZ4 compression failed for HNSW data");
            }

            // Write sizes and compressed data
            uint32_t original_size = static_cast<uint32_t>(file_size);
            uint32_t comp_size = static_cast<uint32_t>(compressed_size);

            output.write(reinterpret_cast<const char*>(&original_size), sizeof(uint32_t));
            output.write(reinterpret_cast<const char*>(&comp_size), sizeof(uint32_t));
            output.write(compressed_data.data(), compressed_size);

            // Clean up
            temp_utils::safe_remove_file(temp_filename);
        }
        catch (...) {
            temp_utils::safe_remove_file(temp_filename);
            throw;
        }
    }

    void load_hnsw_from_stream_compressed(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
        hnswlib::SpaceInterface<float>* space) {
        try {
            // Read sizes
            uint32_t original_size, compressed_size;
            input.read(reinterpret_cast<char*>(&original_size), sizeof(uint32_t));
            input.read(reinterpret_cast<char*>(&compressed_size), sizeof(uint32_t));

            // Read compressed data
            std::vector<char> compressed_data(compressed_size);
            input.read(compressed_data.data(), compressed_size);

            // Decompress with LZ4
            std::vector<char> decompressed_data(original_size);
            int decompressed_size = LZ4_decompress_safe(
                compressed_data.data(), decompressed_data.data(),
                static_cast<int>(compressed_size), static_cast<int>(original_size));

            if (decompressed_size != static_cast<int>(original_size)) {
                throw std::runtime_error("LZ4 decompression failed or size mismatch for HNSW data");
            }

            // Write to temporary file and load
            std::string temp_filename = hnsw_stream_utils::generate_unique_temp_filename("hnsw_decomp");
            std::ofstream temp_file(temp_filename, std::ios::binary);
            if (!temp_file.is_open()) {
                throw std::runtime_error("Failed to create temporary file for HNSW decompression");
            }

            temp_file.write(decompressed_data.data(), original_size);
            temp_file.close();

            // Load from temporary file
            hnsw_index->loadIndex(temp_filename, space);

            // Clean up
            temp_utils::safe_remove_file(temp_filename);
        }
        catch (...) {
            throw;
        }
    }

    // HNSW k-NN query utilities
    void build_knn_graph_hnsw(const std::vector<float>& data, int n_obs, int n_dim, int n_neighbors,
        hnswlib::HierarchicalNSW<float>* hnsw_index, std::vector<int>& nn_indices,
        std::vector<double>& nn_distances) {

        nn_indices.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));
        nn_distances.resize(static_cast<size_t>(n_obs) * static_cast<size_t>(n_neighbors));

        // Use HNSW for fast approximate k-NN queries
#pragma omp parallel for if(n_obs > 1000)
        for (int i = 0; i < n_obs; i++) {
            // Query HNSW index for k+1 neighbors (includes self)
            std::vector<float> query_data(data.begin() + static_cast<size_t>(i) * static_cast<size_t>(n_dim),
                data.begin() + static_cast<size_t>(i + 1) * static_cast<size_t>(n_dim));

            // CRITICAL SAFETY CHECK: Ensure HNSW index is valid
            if (!hnsw_index) {
                continue; // Skip this iteration if no index
            }

            try {
                // Query for k+1 neighbors to account for self-match
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
                    hnsw_index->searchKnn(query_data.data(), n_neighbors + 1);

                // Extract results, skipping self-match
                std::vector<std::pair<float, int>> neighbors;
                neighbors.reserve(n_neighbors + 1);

                while (!result.empty()) {
                    auto& top = result.top();
                    int neighbor_id = static_cast<int>(top.second);
                    if (neighbor_id != i) { // Skip self-match
                        neighbors.emplace_back(top.first, neighbor_id);
                    }
                    result.pop();
                }

                // Reverse to get nearest first, and take only n_neighbors
                std::reverse(neighbors.begin(), neighbors.end());
                if (neighbors.size() > static_cast<size_t>(n_neighbors)) {
                    neighbors.resize(n_neighbors);
                }

                // Fill the arrays
                for (int j = 0; j < n_neighbors && j < static_cast<int>(neighbors.size()); j++) {
                    nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(j)] = neighbors[j].second;
                    nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(j)] = static_cast<double>(neighbors[j].first);
                }

                // Fill remaining slots with -1 and high distance if not enough neighbors
                for (int j = static_cast<int>(neighbors.size()); j < n_neighbors; j++) {
                    nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(j)] = -1;
                    nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(j)] = 1e6;
                }
            }
            catch (...) {
                // Handle HNSW query errors gracefully
                for (int j = 0; j < n_neighbors; j++) {
                    nn_indices[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(j)] = -1;
                    nn_distances[static_cast<size_t>(i) * static_cast<size_t>(n_neighbors) + static_cast<size_t>(j)] = 1e6;
                }
            }
        }
    }

    // HNSW index creation and management
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> create_hnsw_index(
        hnswlib::SpaceInterface<float>* space, int n_obs, int hnsw_M, int hnsw_ef_construction, int hnsw_ef_search) {

        // Memory estimation for HNSW index
        size_t estimated_memory_mb = ((size_t)n_obs * hnsw_M * 4 * 2) / (1024 * 1024);
        printf("[HNSW] Creating index: %d points, estimated %zuMB memory, M=%d, ef_c=%d\n",
            n_obs, estimated_memory_mb, hnsw_M, hnsw_ef_construction);

        auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space, n_obs, hnsw_M, hnsw_ef_construction);
        index->setEf(hnsw_ef_search);  // Set query-time ef parameter

        return index;
    }

    void add_points_to_hnsw(hnswlib::HierarchicalNSW<float>* hnsw_index,
        const std::vector<float>& normalized_data, int n_obs, int n_dim) {

        // Add all points to HNSW index using the normalized data
        for (int i = 0; i < n_obs; i++) {
            hnsw_index->addPoint(
                &normalized_data[static_cast<size_t>(i) * static_cast<size_t>(n_dim)],
                static_cast<hnswlib::labeltype>(i));
        }

        printf("[HNSW] Built index with %d points in %dD space\n", n_obs, n_dim);
    }

    // Temporary normalization utilities (will be moved to separate module later)
    namespace NormalizationPipeline {
        int determine_normalization_mode(UwotMetric metric) {
            // Simple logic for now - will be enhanced in dedicated module
            if (metric == UWOT_METRIC_COSINE || metric == UWOT_METRIC_CORRELATION) {
                return 0; // No z-score normalization for angle-preserving metrics
            }
            return 1; // Use z-score normalization for other metrics
        }

        bool normalize_data_consistent(std::vector<float>& input_data, std::vector<float>& output_data,
            int n_obs, int n_dim, std::vector<float>& means, std::vector<float>& stds, int mode) {

            // Copy input to output
            output_data = input_data;

            // Basic implementation - will be enhanced in dedicated module
            means.assign(n_dim, 0.0f);
            stds.assign(n_dim, 1.0f);

            if (mode == 1) {
                // Basic z-score normalization
                // Compute means
                for (int i = 0; i < n_obs; i++) {
                    for (int j = 0; j < n_dim; j++) {
                        means[j] += input_data[static_cast<size_t>(i) * n_dim + j];
                    }
                }
                for (int j = 0; j < n_dim; j++) {
                    means[j] /= n_obs;
                }

                // Compute stds
                for (int i = 0; i < n_obs; i++) {
                    for (int j = 0; j < n_dim; j++) {
                        float diff = input_data[static_cast<size_t>(i) * n_dim + j] - means[j];
                        stds[j] += diff * diff;
                    }
                }
                for (int j = 0; j < n_dim; j++) {
                    stds[j] = std::sqrt(stds[j] / (n_obs - 1));
                    if (stds[j] < 1e-8f) stds[j] = 1.0f; // Avoid division by zero
                }

                // Apply normalization to output
                for (int i = 0; i < n_obs; i++) {
                    for (int j = 0; j < n_dim; j++) {
                        size_t idx = static_cast<size_t>(i) * n_dim + j;
                        output_data[idx] = (input_data[idx] - means[j]) / stds[j];
                    }
                }
            }

            return true;
        }
    }
}