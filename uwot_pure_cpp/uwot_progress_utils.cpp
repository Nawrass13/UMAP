#include "uwot_progress_utils.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <cstdio>

// Global variables for passing information to v2 callbacks
thread_local float g_current_epoch_loss = 0.0f;
thread_local uwot_progress_callback_v2 g_v2_callback = nullptr;

// Helper functions to send warnings/errors to v2 callback
void send_warning_to_callback(const char* warning_text) {
    if (g_v2_callback) {
        g_v2_callback("Warning", 0, 1, 0.0f, warning_text);
    }
    else {
        // Fallback to console if no callback
        printf("[WARNING] %s\n", warning_text);
    }
}

void send_error_to_callback(const char* error_text) {
    if (g_v2_callback) {
        g_v2_callback("Error", 0, 1, 0.0f, error_text);
    }
    else {
        // Fallback to console if no callback
        printf("[ERROR] %s\n", error_text);
    }
}

// Cross-platform file utilities
namespace temp_utils {
    bool safe_remove_file(const std::string& filename) {
        try {
            return std::filesystem::remove(filename);
        }
        catch (...) {
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
        }
        else if (seconds < 3600) {
            int minutes = static_cast<int>(seconds / 60);
            int secs = static_cast<int>(seconds) % 60;
            oss << minutes << "m " << secs << "s";
        }
        else {
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
            }
            else if (operations > 1e9) { // > 1B operations
                oss << "NOTICE: Large exact k-NN computation (" << n_obs << "x" << n_dim
                    << ") - this may take several minutes.";
            }
        }

        return oss.str();
    }

    // Safe callback invoker - handles null callbacks gracefully
    void safe_callback(uwot_progress_callback_v2 callback,
        const char* phase, int current, int total, float percent,
        const char* message) {
        if (callback != nullptr) {
            callback(phase, current, total, percent, message);
        }
    }
}