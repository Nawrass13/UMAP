#include "uwot_distance.h"

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