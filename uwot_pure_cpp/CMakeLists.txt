cmake_minimum_required(VERSION 3.12)
project(UMAPWrapper VERSION 2.0.0)

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
    add_executable(test_enhanced_wrapper
        ${CMAKE_CURRENT_SOURCE_DIR}/enhanced_test.cpp
    )
    target_link_libraries(test_enhanced_wrapper uwot_wrapper)


    # Suppress warnings for test executables
    if(MSVC)
        target_compile_options(test_enhanced_wrapper PRIVATE
            /wd4244  # conversion warnings
            /wd4267  # size_t conversion warnings
            /wd4566  # unicode character warnings
        )
    endif()

    # Add tests
    enable_testing()
    add_test(NAME EnhancedWrapperTest COMMAND test_enhanced_wrapper)
endif()

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/UMAPWrapperConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Export targets for build tree
export(TARGETS uwot_wrapper
    FILE "${CMAKE_CURRENT_BINARY_DIR}/UMAPWrapperTargets.cmake"
)

# Install the export set
install(EXPORT UMAPWrapperTargets
    FILE UMAPWrapperTargets.cmake
    DESTINATION lib/cmake/UMAPWrapper
)