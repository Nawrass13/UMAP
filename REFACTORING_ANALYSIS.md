# UMAP C++ Refactoring Functional Analysis

## Current State Analysis (2,381 lines)

### 📊 Line Distribution by Function:
- **Core Model Structure**: ~150 lines (UwotModel struct + basic functions)
- **Persistence (Save/Load)**: ~450 lines (18.9% of codebase)
- **Fit/Training**: ~600 lines (25.2% of codebase)
- **Transform/Projection**: ~400 lines (16.8% of codebase)
- **Distance Metrics**: ~300 lines (12.6% of codebase)
- **k-NN Graph Building**: ~200 lines (8.4% of codebase)
- **Product Quantization**: ~150 lines (6.3% of codebase)
- **Utility Functions**: ~121 lines (5.1% of codebase)

## 🎯 Primary Functional Modules (Your Preferences)

### 1. **PERSISTENCE MODULE** 🗄️ Priority: HIGHEST
**File**: `uwot_persistence.h/.cpp`
**Lines**: ~450 lines (largest single module)
**Functions**:
- `uwot_save_model()` (lines 1885-2038, ~153 lines)
- `uwot_load_model()` (lines 2039-2294, ~255 lines)
- HNSW stream operations (lines 707-794, ~87 lines)
- LZ4 compression utilities

**Key Features**:
- Complete model serialization/deserialization
- HNSW index persistence with compression
- k-NN fallback data for exact reproducibility
- Cross-platform file handling
- Error recovery and validation

**Dependencies**: Core Model, HNSW Utils, Progress Utils

---

### 2. **FIT/TRAINING MODULE** 🏋️ Priority: HIGH
**File**: `uwot_fit.h/.cpp`
**Lines**: ~600 lines (second largest)
**Functions**:
- `uwot_fit_with_progress()` (lines 1029-1392, ~363 lines)
- `uwot_fit_with_progress_v2()` (lines 1393-1462, ~69 lines)
- Training pipeline coordination
- HNSW index building during training
- Normalization integration

**Key Features**:
- Complete UMAP training algorithm
- HNSW-accelerated k-NN computation
- Progress reporting with v1/v2 callback systems
- Data normalization pipeline
- Hyperparameter optimization

**Dependencies**: Core Model, Distance Metrics, k-NN Graph, HNSW Utils, Normalization

---

### 3. **PROJECTION/TRANSFORM MODULE** 🎯 Priority: HIGH
**File**: `uwot_transform.h/.cpp`
**Lines**: ~400 lines
**Functions**:
- `uwot_transform()` (lines 1471-1654, ~183 lines)
- `uwot_transform_detailed()` (lines 1655-1884, ~229 lines)
- Safety analysis and outlier detection
- Transform data normalization

**Key Features**:
- High-speed HNSW-accelerated transforms (<3ms)
- 5-level outlier detection (Normal → No Man's Land)
- Confidence scoring and safety metrics
- Percentile ranking and z-score analysis
- Production-ready data validation

**Dependencies**: Core Model, HNSW Utils, Distance Metrics

---

### 4. **PRODUCT QUANTIZATION MODULE** 📦 Priority: MEDIUM
**File**: `uwot_quantization.h/.cpp`
**Lines**: ~150 lines
**Functions**:
- PQ encoding/decoding (lines 200-258, ~58 lines)
- PQ utilities namespace (lines 112-199, ~87 lines)
- Memory optimization features

**Key Features**:
- Product Quantization for memory reduction
- Codebook management
- Vector encoding/decoding
- Memory-efficient storage

**Dependencies**: Core Model, Distance Metrics

**Note**: Currently present but can be deprioritized per project evolution

---

## 🔧 Supporting Modules (Infrastructure)

### 5. **CORE MODEL MODULE** 🏗️ Priority: CRITICAL
**File**: `uwot_model.h/.cpp`
**Lines**: ~200 lines
**Functions**:
- `uwot_create()`, `uwot_destroy()`
- `uwot_get_model_info()`, info functions
- UwotModel struct definition (lines 39-109, ~70 lines)

**Key Features**:
- Central model structure definition
- Model lifecycle management
- HNSW integration points
- Parameter validation

**Dependencies**: None (foundational)

---

### 6. **DISTANCE METRICS MODULE** 📏 Priority: HIGH
**File**: `uwot_distance.h/.cpp`
**Lines**: ~300 lines
**Functions**:
- Distance metric implementations (lines 403-495, ~92 lines)
- Metric-specific optimizations
- HNSW space selection logic

**Key Features**:
- Euclidean, Cosine, Manhattan, Correlation, Hamming
- Optimized distance computations
- HNSW compatibility matrix
- Performance-critical code

**Dependencies**: Core Model

---

### 7. **k-NN GRAPH MODULE** 🕸️ Priority: MEDIUM
**File**: `uwot_knn.h/.cpp`
**Lines**: ~200 lines
**Functions**:
- `build_knn_graph()` (lines 496-670, ~174 lines)
- Edge list conversion (lines 671-705, ~34 lines)
- Neighbor statistics computation

**Key Features**:
- HNSW-accelerated k-NN (50-2000x speedup)
- Exact k-NN fallback for unsupported metrics
- Graph structure building for UMAP algorithm

**Dependencies**: Core Model, Distance Metrics, HNSW Utils

---

### 8. **NORMALIZATION MODULE** 🎚️ Priority: MEDIUM
**File**: `uwot_normalization.h/.cpp`
**Lines**: ~150 lines
**Functions**:
- Normalization utilities (lines 259-302, ~43 lines)
- Metric-specific normalization logic
- Z-score and unit normalization

**Key Features**:
- Data preprocessing for optimal UMAP results
- Metric-aware normalization strategies
- Cosine distance unit normalization
- Statistical parameter computation

**Dependencies**: Core Model, Progress Utils

---

## 📈 Extraction Impact Analysis

### Module Extraction Priority (Based on Size + Complexity):

1. **Persistence (450 lines)** → 19% reduction ⭐⭐⭐⭐⭐
2. **Fit/Training (600 lines)** → 25% reduction ⭐⭐⭐⭐⭐
3. **Transform (400 lines)** → 17% reduction ⭐⭐⭐⭐
4. **Distance Metrics (300 lines)** → 13% reduction ⭐⭐⭐
5. **k-NN Graph (200 lines)** → 8% reduction ⭐⭐
6. **Core Model (200 lines)** → 8% reduction ⭐⭐
7. **Quantization (150 lines)** → 6% reduction ⭐
8. **Normalization (150 lines)** → 6% reduction ⭐

### Expected Results After Full Extraction:
- **Main wrapper**: 2,381 → ~200 lines (92% reduction)
- **Module count**: 8 focused files + 2 existing (progress, hnsw)
- **Average module size**: 150-600 lines each
- **Maintainability**: Dramatically improved

## 🚀 Recommended Extraction Sequence

### Phase 2: Core Infrastructure
1. **Core Model** → Foundation for all other modules
2. **Distance Metrics** → Needed by most other modules

### Phase 3: Major Functionality (Your Priorities)
3. **Persistence** → Largest single module (450 lines)
4. **Fit/Training** → Second largest (600 lines)
5. **Transform/Projection** → Third largest (400 lines)

### Phase 4: Specialized Features
6. **k-NN Graph** → Algorithm-specific
7. **Normalization** → Data preprocessing
8. **Quantization** → Memory optimization (optional)

## 🎯 Immediate Next Steps

1. **Extract Core Model** - Enables all other extractions
2. **Extract Persistence** - Your highest priority, largest impact
3. **Extract Fit/Training** - Your second priority, massive complexity reduction
4. **Extract Transform** - Your third priority, completes core functionality

This sequence aligns with your preferences (persistence, fit, projection) while ensuring proper dependency management and maximum impact.

---

**Analysis Complete**: Ready for systematic functional extraction based on your priorities! 🎉