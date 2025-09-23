#pragma once

#include "uwot_model.h"
#include "uwot_hnsw_utils.h"
#include "uwot_progress_utils.h"
#include <iostream>
#include <fstream>

namespace persistence_utils {

    // Main persistence functions
    int save_model(UwotModel* model, const char* filename);
    UwotModel* load_model(const char* filename);

    // HNSW compression utilities
    void save_hnsw_to_stream_compressed(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index);
    void load_hnsw_from_stream_compressed(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
        hnswlib::SpaceInterface<float>* space);

    // Model state debugging
    void debug_print_model_state(UwotModel* model, const char* context);
}