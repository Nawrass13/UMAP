# UMAP C++ Refactoring Plan - Session State

## Current Status (2025-09-22)
- **Original size**: 2,865 lines in uwot_simple_wrapper.cpp
- **Current size**: 2,381 lines (384 lines extracted)
- **Progress utilities**: ✅ COMPLETED - Extracted to uwot_progress_utils.h/.cpp
- **HNSW utilities**: 🔄 IN PROGRESS - Partially extracted, needs compilation fixes

## Functional Module Extraction Plan

### Phase 1: Core Infrastructure ✅🔄
1. **Progress Utilities** ✅ COMPLETED
   - File: `uwot_progress_utils.h/.cpp`
   - Content: Global callbacks, error/warning handlers, time formatting, temp file utils
   - Status: Fully extracted and working

2. **HNSW Utilities** 🔄 NEEDS COMPLETION
   - File: `uwot_hnsw_utils.h/.cpp`
   - Content: L1Space, SpaceFactory, HNSW index management, compression
   - Status: Created but missing normalization features causing compilation errors
   - **IMMEDIATE FIX NEEDED**: Add missing NormalizationPipeline to fix compilation

### Phase 2: Core Model Structure
3. **Core Model** (Priority: HIGH)
   - File: `uwot_model.h/.cpp`
   - Content: UwotModel struct definition, model creation/destruction
   - Functions: `uwot_create()`, `uwot_destroy()`, model field management
   - Lines to extract: ~100-150 lines

4. **Distance Metrics** (Priority: HIGH)
   - File: `uwot_distance.h/.cpp`
   - Content: All distance computation functions (Euclidean, Cosine, Manhattan, etc.)
   - Functions: Distance metric implementations, metric selection logic
   - Lines to extract: ~200-300 lines

### Phase 3: Core Algorithms
5. **k-NN Graph Building** (Priority: HIGH)
   - File: `uwot_knn.h/.cpp`
   - Content: Neighbor graph construction, both exact and HNSW-accelerated
   - Functions: `build_knn_graph()`, neighbor statistics computation
   - Lines to extract: ~300-400 lines

6. **Normalization Pipeline** (Priority: HIGH)
   - File: `uwot_normalization.h/.cpp`
   - Content: Data normalization, z-score, unit normalization, metric-specific logic
   - Functions: Normalization utilities currently missing from HNSW extraction
   - Lines to extract: ~200-250 lines

### Phase 4: Main UMAP Functions
7. **Fitting/Training** (Priority: MEDIUM)
   - File: `uwot_fit.h/.cpp`
   - Content: All `uwot_fit_*` functions
   - Functions: `uwot_fit()`, `uwot_fit_with_progress()`, `uwot_fit_with_enhanced_progress()`, `uwot_fit_with_progress_v2()`
   - Lines to extract: ~400-500 lines

8. **Transform** (Priority: MEDIUM)
   - File: `uwot_transform.h/.cpp`
   - Content: All `uwot_transform_*` functions
   - Functions: `uwot_transform()`, `uwot_transform_detailed()`, safety analysis
   - Lines to extract: ~300-400 lines

9. **Persistence** (Priority: MEDIUM)
   - File: `uwot_persistence.h/.cpp`
   - Content: Model save/load functionality
   - Functions: `uwot_save_model()`, `uwot_load_model()`, serialization
   - Lines to extract: ~400-500 lines

### Phase 5: Utility Functions
10. **Model Information** (Priority: LOW)
    - File: `uwot_info.h/.cpp`
    - Content: Model introspection and information functions
    - Functions: `uwot_get_model_info()`, version management
    - Lines to extract: ~100-150 lines

## Expected Results After Full Refactoring

### File Structure
```
uwot_pure_cpp/
├── uwot_simple_wrapper.cpp    # Main API entry points (~200-300 lines)
├── uwot_simple_wrapper.h      # Public API declarations
├── uwot_progress_utils.h/.cpp  # ✅ Progress & error handling
├── uwot_hnsw_utils.h/.cpp      # 🔄 HNSW optimization
├── uwot_model.h/.cpp           # Core model structure
├── uwot_distance.h/.cpp        # Distance metrics
├── uwot_knn.h/.cpp             # k-NN graph building
├── uwot_normalization.h/.cpp   # Data normalization
├── uwot_fit.h/.cpp             # Training functions
├── uwot_transform.h/.cpp       # Transform functions
├── uwot_persistence.h/.cpp     # Save/load functions
├── uwot_info.h/.cpp            # Model information
└── CMakeLists.txt              # Updated build system
```

### Size Reduction
- **Main wrapper**: From 2,865 lines → ~200-300 lines (90% reduction)
- **Total modules**: 10-12 focused, single-responsibility files
- **Average module size**: 150-400 lines each
- **Maintainability**: Dramatically improved

## Immediate Next Steps

### URGENT: Fix Compilation (Session Priority)
1. **Add missing normalization to HNSW module** or create separate normalization module
2. **Fix UwotModel struct** - add missing `normalization_mode` field
3. **Test compilation** after each fix
4. **Commit working state** before continuing

### Next Session Continuation
1. **Complete HNSW utilities** extraction properly
2. **Extract Core Model** (UwotModel struct) - highest impact
3. **Extract Distance Metrics** - second highest impact
4. **Continue systematically** through the plan

## Module Dependencies (Build Order)
1. `uwot_progress_utils` ✅ (no dependencies)
2. `uwot_model` (depends on: progress_utils)
3. `uwot_distance` (depends on: model, progress_utils)
4. `uwot_normalization` (depends on: model, progress_utils)
5. `uwot_hnsw_utils` (depends on: model, distance, normalization)
6. `uwot_knn` (depends on: model, distance, hnsw_utils)
7. `uwot_fit` (depends on: model, knn, normalization, hnsw_utils)
8. `uwot_transform` (depends on: model, hnsw_utils, normalization)
9. `uwot_persistence` (depends on: model, hnsw_utils)
10. `uwot_info` (depends on: model)

## Success Metrics
- ✅ **Compilation**: All modules compile without errors
- ✅ **Functionality**: All existing tests pass (15/15)
- ✅ **Performance**: No regressions in HNSW optimization
- ✅ **Maintainability**: Single-responsibility, focused modules
- ✅ **Extensibility**: Easy to add new features to specific modules

---
**Session State Saved**: 2025-09-22
**Next Priority**: Fix compilation issues, then continue systematic extraction