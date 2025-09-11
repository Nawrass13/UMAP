#ifndef UWOT_SIMPLE_WRAPPER_H
#define UWOT_SIMPLE_WRAPPER_H

// DLL export/import macros for Windows
#ifdef _WIN32
#ifdef UWOT_STATIC
#define UWOT_API
#elif defined(UWOT_EXPORTS)
#define UWOT_API __declspec(dllexport)
#else
#define UWOT_API __declspec(dllimport)
#endif
#else
#define UWOT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // Opaque pointer to the UMAP model
    typedef struct UwotModel UwotModel;

    // Error codes
#define UWOT_SUCCESS 0
#define UWOT_ERROR_INVALID_PARAMS -1
#define UWOT_ERROR_MEMORY -2
#define UWOT_ERROR_NOT_IMPLEMENTED -3
#define UWOT_ERROR_FILE_IO -4
#define UWOT_ERROR_MODEL_NOT_FITTED -5
#define UWOT_ERROR_INVALID_MODEL_FILE -6

// Basic UMAP operations
    UWOT_API UwotModel* uwot_create();

    UWOT_API int uwot_fit(UwotModel* model,
        float* data,
        int n_obs,
        int n_dim,
        int n_neighbors,
        float min_dist,
        int n_epochs,
        float* embedding);

    UWOT_API int uwot_transform(UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding);

    UWOT_API void uwot_destroy(UwotModel* model);

    // Model persistence functions
    UWOT_API int uwot_save_model(UwotModel* model, const char* filename);

    UWOT_API UwotModel* uwot_load_model(const char* filename);

    UWOT_API int uwot_get_model_info(UwotModel* model,
        int* n_vertices,
        int* n_dim,
        int* embedding_dim,
        int* n_neighbors,
        float* min_dist);

    // Additional utility functions
    UWOT_API const char* uwot_get_error_message(int error_code);

    UWOT_API int uwot_get_embedding_dim(UwotModel* model);

    UWOT_API int uwot_get_n_vertices(UwotModel* model);

    UWOT_API int uwot_is_fitted(UwotModel* model);

#ifdef __cplusplus
}
#endif

#endif // UWOT_SIMPLE_WRAPPER_H