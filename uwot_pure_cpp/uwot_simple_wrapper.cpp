#include "uwot_simple_wrapper.h"
#include "uwot_progress_utils.h"
#include "uwot_hnsw_utils.h"
#include "uwot_model.h"
#include "uwot_persistence.h"
#include "uwot_fit.h"
#include "uwot_transform.h"
#include "uwot_quantization.h"

#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iostream>

extern "C" {

    UWOT_API UwotModel* uwot_create() {
        return model_utils::create_model();
    }

    UWOT_API void uwot_destroy(UwotModel* model) {
        model_utils::destroy_model(model);
    }

    UWOT_API int uwot_fit_with_progress(
        UwotModel* model,
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
        int force_exact_knn,
        int M,
        int ef_construction,
        int ef_search,
        int use_quantization
    ) {
        return fit_utils::uwot_fit_with_progress(model, data, n_obs, n_dim, embedding_dim, n_neighbors,
            min_dist, spread, n_epochs, metric, embedding, progress_callback,
            force_exact_knn, M, ef_construction, ef_search, use_quantization);
    }

    UWOT_API int uwot_fit_with_progress_v2(
        UwotModel* model,
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
        int force_exact_knn,
        int M,
        int ef_construction,
        int ef_search,
        int use_quantization
    ) {
        return fit_utils::uwot_fit_with_progress_v2(model, data, n_obs, n_dim, embedding_dim, n_neighbors,
            min_dist, spread, n_epochs, metric, embedding, progress_callback,
            force_exact_knn, M, ef_construction, ef_search, use_quantization);
    }

    UWOT_API int uwot_transform(
        UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding
    ) {
        return transform_utils::uwot_transform(model, new_data, n_new_obs, n_dim, embedding);
    }

    UWOT_API int uwot_transform_detailed(
        UwotModel* model,
        float* new_data,
        int n_new_obs,
        int n_dim,
        float* embedding,
        int* nn_indices,
        float* nn_distances,
        float* confidence_score,
        int* outlier_level,
        float* percentile_rank,
        float* z_score
    ) {
        return transform_utils::uwot_transform_detailed(model, new_data, n_new_obs, n_dim, embedding,
            nn_indices, nn_distances, confidence_score, outlier_level, percentile_rank, z_score);
    }

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
        int* hnsw_ef_search) {
        if (!model) {
            return UWOT_ERROR_INVALID_PARAMS;
        }

        if (n_vertices) *n_vertices = model->n_vertices;
        if (n_dim) *n_dim = model->n_dim;
        if (embedding_dim) *embedding_dim = model->embedding_dim;
        if (n_neighbors) *n_neighbors = model->n_neighbors;
        if (min_dist) *min_dist = model->min_dist;
        if (spread) *spread = model->spread;
        if (metric) *metric = model->metric;
        if (hnsw_M) *hnsw_M = model->hnsw_M;
        if (hnsw_ef_construction) *hnsw_ef_construction = model->hnsw_ef_construction;
        if (hnsw_ef_search) *hnsw_ef_search = model->hnsw_ef_search;

        return UWOT_SUCCESS;
    }

    UWOT_API int uwot_save_model(UwotModel* model, const char* filename) {
        return persistence_utils::save_model(model, filename);
    }

    UWOT_API UwotModel* uwot_load_model(const char* filename) {
        return persistence_utils::load_model(filename);
    }

    UWOT_API const char* uwot_get_error_message(int error_code) {
        return model_utils::get_error_message(error_code);
    }

    UWOT_API const char* uwot_get_metric_name(UwotMetric metric) {
        return model_utils::get_metric_name(metric);
    }

    UWOT_API int uwot_get_embedding_dim(UwotModel* model) {
        return model_utils::get_embedding_dim(model);
    }

    UWOT_API int uwot_get_n_vertices(UwotModel* model) {
        return model_utils::get_n_vertices(model);
    }

    UWOT_API int uwot_is_fitted(UwotModel* model) {
        return model_utils::is_fitted(model);
    }

    UWOT_API const char* uwot_get_version() {
        return model_utils::get_version();
    }

    UWOT_API void uwot_set_global_callback(uwot_progress_callback_v2 callback) {
        g_v2_callback = callback;
    }

    UWOT_API void uwot_clear_global_callback() {
        g_v2_callback = nullptr;
    }

} // extern "C"