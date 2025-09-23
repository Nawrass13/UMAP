#include "uwot_model.h"
#include "uwot_progress_utils.h"

namespace model_utils {

    UwotModel* create_model() {
        try {
            UwotModel* model = new UwotModel();
            return model;
        }
        catch (...) {
            return nullptr;
        }
    }

    void destroy_model(UwotModel* model) {
        if (model) {
            delete model;
        }
    }

    int get_model_info(UwotModel* model, int* n_vertices, int* n_dim, int* embedding_dim,
        int* n_neighbors, float* min_dist, float* spread, UwotMetric* metric,
        int* hnsw_M, int* hnsw_ef_construction, int* hnsw_ef_search) {

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

    int get_embedding_dim(UwotModel* model) {
        if (!model) {
            return -1;
        }
        return model->embedding_dim;
    }

    int get_n_vertices(UwotModel* model) {
        if (!model) {
            return -1;
        }
        return model->n_vertices;
    }

    int is_fitted(UwotModel* model) {
        if (!model) {
            return 0;
        }
        return model->is_fitted ? 1 : 0;
    }

    const char* get_error_message(int error_code) {
        switch (error_code) {
        case UWOT_SUCCESS:
            return "Success";
        case UWOT_ERROR_INVALID_PARAMS:
            return "Invalid parameters provided";
        case UWOT_ERROR_MEMORY:
            return "Memory allocation failed";
        case UWOT_ERROR_NOT_IMPLEMENTED:
            return "Feature not implemented";
        case UWOT_ERROR_FILE_IO:
            return "File I/O operation failed";
        case UWOT_ERROR_MODEL_NOT_FITTED:
            return "Model has not been fitted yet";
        case UWOT_ERROR_INVALID_MODEL_FILE:
            return "Invalid model file";
        default:
            return "Unknown error";
        }
    }

    const char* get_metric_name(UwotMetric metric) {
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

    const char* get_version() {
        return UWOT_WRAPPER_VERSION_STRING;
    }
}