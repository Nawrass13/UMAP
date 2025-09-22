cmake_minimum_required(VERSION 3.12)
project(UMWWGBWrapper VERSION 3.5.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" ON)

# Find dependencies
find_package(OpenMP)

# Include directories (current directory since all files are in root)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files (using actual file names in root directory)
set(UWOT_WRAPPER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/uwot_simple_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lz4.c
)

# Core headers (all in root directory)
set(UWOT_CORE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/smooth_knn.h
    ${CMAKE_CURRENT_SOURCE_DIR}/optimize.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gradient.h
    ${CMAKE_CURRENT_SOURCE_DIR}/transform.h
    ${CMAKE_CURRENT_SOURCE_DIR}/update.h
    ${CMAKE_CURRENT_SOURCE_DIR}/epoch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/perplexity.h
    ${CMAKE_CURRENT_SOURCE_DIR}/coords.h
    ${CMAKE_CURRENT_SOURCE_DIR}/connected_components.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tauprng.h
    ${CMAKE_CURRENT_SOURCE_DIR}/sampler.h
    ${CMAKE_CURRENT_SOURCE_DIR}/supervised.h
)

# Create the library
add_library(uwot_wrapper ${UWOT_WRAPPER_SOURCES})

# Link OpenMP AFTER target is created
if(OpenMP_CXX_FOUND)
    target_link_libraries(uwot_wrapper PUBLIC OpenMP::OpenMP_CXX)
endif()

# Set library properties
set_target_properties(uwot_wrapper PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/uwot_simple_wrapper.h"
)

# Compile definitions
target_compile_definitions(uwot_wrapper PRIVATE UWOT_EXPORTS)

# Suppress warnings from third-party uwot headers
if(MSVC)
    target_compile_options(uwot_wrapper PRIVATE 
        /wd4244  # conversion from 'double' to 'float', possible loss of data
        /wd4267  # conversion from 'size_t' to 'int', possible loss of data
        /wd4566  # character represented by universal-character-name cannot be represented
    )
endif()

# Link libraries
target_link_libraries(uwot_wrapper PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# Platform-specific settings
if(WIN32)
    set_target_properties(uwot_wrapper PROPERTIES
        OUTPUT_NAME "uwot"
        SUFFIX ".dll"
    )
    target_compile_definitions(uwot_wrapper PRIVATE WIN32_WINNT=0x0601)
else()
    set_target_properties(uwot_wrapper PROPERTIES
        OUTPUT_NAME "uwot"
        PREFIX "lib"
        SUFFIX ".so"
    )
endif()

# Install targets and create install tree export
install(TARGETS uwot_wrapper
    EXPORT UMAPWrapperTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)

# Install uwot headers
install(FILES ${UWOT_CORE_HEADERS}
    DESTINATION include/uwot
)

# Tests
if(BUILD_TESTS)
    add_executable(simple_save_load_test ${CMAKE_CURRENT_SOURCE_DIR}/simple_save_load_test.cpp)
    target_link_libraries(simple_save_load_test uwot_wrapper)

    add_executable(debug_large_save_load ${CMAKE_CURRENT_SOURCE_DIR}/debug_large_save_load.cpp)
    target_link_libraries(debug_large_save_load uwot_wrapper)

    add_executable(test_fixed_large_scale ${CMAKE_CURRENT_SOURCE_DIR}/test_fixed_large_scale.cpp)
    target_link_libraries(test_fixed_large_scale uwot_wrapper)

    add_executable(test_user_params ${CMAKE_CURRENT_SOURCE_DIR}/test_user_params.cpp)
    target_link_libraries(test_user_params uwot_wrapper)

    add_executable(test_user_exact_params ${CMAKE_CURRENT_SOURCE_DIR}/test_user_exact_params.cpp)
    target_link_libraries(test_user_exact_params uwot_wrapper)

    add_executable(test_hnsw_exact_mse_unified ${CMAKE_CURRENT_SOURCE_DIR}/test_hnsw_exact_mse_unified.cpp)
    target_link_libraries(test_hnsw_exact_mse_unified uwot_wrapper)

    add_executable(test_realistic_mse ${CMAKE_CURRENT_SOURCE_DIR}/test_realistic_mse.cpp)
    target_link_libraries(test_realistic_mse uwot_wrapper)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(simple_save_load_test OpenMP::OpenMP_CXX)
        target_link_libraries(debug_large_save_load OpenMP::OpenMP_CXX)
        target_link_libraries(test_fixed_large_scale OpenMP::OpenMP_CXX)
        target_link_libraries(test_user_params OpenMP::OpenMP_CXX)
        target_link_libraries(test_user_exact_params OpenMP::OpenMP_CXX)
        target_link_libraries(test_hnsw_exact_mse_unified OpenMP::OpenMP_CXX)
        target_link_libraries(test_realistic_mse OpenMP::OpenMP_CXX)
    endif()
endif()

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/UMWWGBWrapperConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Export targets for build tree
export(TARGETS uwot_wrapper
    FILE "${CMAKE_CURRENT_BINARY_DIR}/UMWWGBWrapperTargets.cmake"
)

# Install the export set
install(EXPORT UMAPWrapperTargets
    FILE UMAPWrapperTargets.cmake
    DESTINATION lib/cmake/UMWWGBWrapper
)