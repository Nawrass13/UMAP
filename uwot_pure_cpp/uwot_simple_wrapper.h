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

// Distance metrics
    typedef enum {
        UWOT_METRIC_EUCLIDEAN = 0,
        UWOT_METRIC_COSINE = 1,
        UWOT_METRIC_MANHATTAN = 2,
        UWOT_METRIC_CORRELATION = 3,
        UWOT_METRIC_HAMMING = 4
    } UwotMetric;

    // Forward declaration
    typedef struct UwotModel UwotModel;

    // Progress callback function type
    typedef void (*uwot_progress_callback)(int epoch, int total_epochs, float percent);

    // Core functions
    UWOT_API UwotModel* uwot_create();
    UWOT_API void uwot_destroy(UwotModel* model);

    // Training functions
    UWOT_API int uwot_fit(UwotModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float min_dist,
        int n_epochs,
        UwotMetric metric,
        float* embedding);

    UWOT_API int uwot_fit_with_progress(UwotModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int embedding_dim,
        int n_neighbors,
        float min_dist,
        int n_epochs,
        UwotMetric metric,
        float* embedding,
        uwot_progress_callback progress_callback);

    // Transform function (out-of-sample projection)
    UWOT_API int uwot_transform(UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding);

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
        UwotMetric* metric);

    // Utility functions
    UWOT_API const char* uwot_get_error_message(int error_code);
    UWOT_API const char* uwot_get_metric_name(UwotMetric metric);
    UWOT_API int uwot_get_embedding_dim(UwotModel* model);
    UWOT_API int uwot_get_n_vertices(UwotModel* model);
    UWOT_API int uwot_is_fitted(UwotModel* model);

#ifdef __cplusplus
}
#endif

#endif // UWOT_SIMPLE_WRAPPER_H