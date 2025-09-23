#pragma once

#include "uwot_model.h"
#include <cmath>
#include <algorithm>

// Distance metric implementations for UMAP
namespace distance_metrics {

    // Individual distance metric functions
    float euclidean_distance(const float* a, const float* b, int dim);
    float cosine_distance(const float* a, const float* b, int dim);
    float manhattan_distance(const float* a, const float* b, int dim);
    float correlation_distance(const float* a, const float* b, int dim);
    float hamming_distance(const float* a, const float* b, int dim);

    // Unified distance computation based on metric type
    float compute_distance(const float* a, const float* b, int dim, UwotMetric metric);

}