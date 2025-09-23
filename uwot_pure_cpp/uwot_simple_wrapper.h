#ifndef UWOT_SIMPLE_WRAPPER_H
#define UWOT_SIMPLE_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

    // Export macros
#ifdef _WIN32
#ifdef UWOT_EXPORTS
#define UWOT_API __declspec(dllexport)
#else
#define UWOT_API __declspec(dllimport)
#endif
#else
#define UWOT_API __attribute__((visibility("default")))
#endif

// Error codes
#define UWOT_SUCCESS 0
#define UWOT_ERROR_INVALID_PARAMS -1
#define UWOT_ERROR_MEMORY -2
#define UWOT_ERROR_NOT_IMPLEMENTED -3
#define UWOT_ERROR_FILE_IO -4
#define UWOT_ERROR_MODEL_NOT_FITTED -5
#define UWOT_ERROR_INVALID_MODEL_FILE -6

// Version information
#define UWOT_WRAPPER_VERSION_STRING "3.11.0"

// Distance metrics
    typedef enum {
        UWOT_METRIC_EUCLIDEAN = 0,
        UWOT_METRIC_COSINE = 1,
        UWOT_METRIC_MANHATTAN = 2,
        UWOT_METRIC_CORRELATION = 3,
        UWOT_METRIC_HAMMING = 4
    } UwotMetric;

    // Outlier level enumeration for enhanced safety detection
    typedef enum {
        UWOT_OUTLIER_NORMAL = 0,      // Within normal range (≤ p95)
        UWOT_OUTLIER_UNUSUAL = 1,     // Unusual but acceptable (p95-p99)
        UWOT_OUTLIER_MILD = 2,        // Mild outlier (p99 to 2.5σ)
        UWOT_OUTLIER_EXTREME = 3,     // Extreme outlier (2.5σ to 4σ)
        UWOT_OUTLIER_NOMANSLAND = 4   // No man's land (> 4σ)
    } UwotOutlierLevel;

    // Forward declaration
    typedef struct UwotModel UwotModel;

    // Progress callback function types
    typedef void (*uwot_progress_callback)(int epoch, int total_epochs, float percent);

    // Enhanced progress callback with phase information, time estimates, and warnings
    typedef void (*uwot_progress_callback_v2)(
        const char* phase,        // Current phase: "Normalizing", "Building HNSW", "k-NN Graph", etc.
        int current,              // Current progress counter
        int total,                // Total items to process
        float percent,            // Progress percentage (0-100)
        const char* message       // Time estimates, warnings, or NULL for no message
    );

    // Core functions
    UWOT_API UwotModel* uwot_create();
    UWOT_API void uwot_destroy(UwotModel* model);

    // UNIFIED TRAINING PIPELINE - All functions use same core implementation
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
        int force_exact_knn = 0,
        int M = -1,
        int ef_construction = -1,
        int ef_search = -1);

    UWOT_API int uwot_fit_with_progress_v2(UwotModel* model,
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
        int force_exact_knn = 0,
        int M = -1,
        int ef_construction = -1,
        int ef_search = -1);

    // Global callback management functions
    UWOT_API void uwot_set_global_callback(uwot_progress_callback_v2 callback);
    UWOT_API void uwot_clear_global_callback();

    // Transform functions
    UWOT_API int uwot_transform(UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding);

    // Enhanced transform function with comprehensive safety analysis
    // Returns detailed information about nearest neighbors, confidence, and outlier detection
    // Parameters:
    //   - embedding: Output embedding coordinates [n_new_obs * embedding_dim]
    //   - nn_indices: Output nearest neighbor indices [n_new_obs * n_neighbors] (can be NULL)
    //   - nn_distances: Output nearest neighbor distances [n_new_obs * n_neighbors] (can be NULL)  
    //   - confidence_score: Output confidence scores [n_new_obs] (0.0-1.0, can be NULL)
    //   - outlier_level: Output outlier levels [n_new_obs] (UwotOutlierLevel enum, can be NULL)
    //   - percentile_rank: Output percentile ranks [n_new_obs] (0-100, can be NULL)
    //   - z_score: Output z-scores [n_new_obs] (standard deviations from mean, can be NULL)
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
        float* z_score);

    // Model persistence
    UWOT_API int uwot_save_model(UwotModel* model, const char* filename);
    UWOT_API UwotModel* uwot_load_model(const char* filename);

    // Model information
    UWOT_API int uwot_get_model_info(UwotModel* model,
        int* n_vertices,
        int* n_dim,
        int* embedding_dim,
        int* n_neighbors,
        float* min_dist,
        float* spread,
        UwotMetric* metric,
        int* hnsw_M,
        int* hnsw_ef_construction,
        int* hnsw_ef_search);

    // Utility functions
    UWOT_API const char* uwot_get_error_message(int error_code);
    UWOT_API const char* uwot_get_metric_name(UwotMetric metric);
    UWOT_API int uwot_get_embedding_dim(UwotModel* model);
    UWOT_API int uwot_get_n_vertices(UwotModel* model);
    UWOT_API int uwot_is_fitted(UwotModel* model);
    UWOT_API const char* uwot_get_version();

#ifdef __cplusplus
}
#endif

#endif // UWOT_SIMPLE_WRAPPER_H