#pragma once

#include "uwot_simple_wrapper.h"
#include <string>

// Global variables for passing information to v2 callbacks
extern thread_local float g_current_epoch_loss;
extern thread_local uwot_progress_callback_v2 g_v2_callback;

// Helper functions to send warnings/errors to v2 callback
void send_warning_to_callback(const char* warning_text);
void send_error_to_callback(const char* error_text);

// Cross-platform file utilities
namespace temp_utils {
    bool safe_remove_file(const std::string& filename);
}

// Enhanced progress reporting utilities
namespace progress_utils {
    // Format time duration in human-readable format
    std::string format_duration(double seconds);

    // Estimate remaining time based on current progress
    std::string estimate_remaining_time(int current, int total, double elapsed_seconds);

    // Generate complexity-based time warnings
    std::string generate_complexity_warning(int n_obs, int n_dim, const std::string& operation);

    // Safe callback invoker - handles null callbacks gracefully
    void safe_callback(uwot_progress_callback_v2 callback,
        const char* phase, int current, int total, float percent,
        const char* message = nullptr);
}