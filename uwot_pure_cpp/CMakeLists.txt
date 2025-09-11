cmake_minimum_required(VERSION 3.16)
project(uwot_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add current directory to include path
include_directories(.)

# Platform-specific compiler settings
if(MSVC)
    # Windows MSVC settings
    add_compile_options(/wd4244)  # Suppress C4244 warnings
    add_compile_options(/wd4101)  # Suppress C4101 warnings (unreferenced local variable)
    add_compile_options(/W3)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Linux GCC/Clang settings
    add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-unused-variable)
    add_compile_options(-fPIC)  # Position independent code for shared library
endif()

# Try to find OpenMP
find_package(OpenMP QUIET)

# Build shared library (.dll on Windows, .so on Linux)
add_library(uwot_shared SHARED uwot_simple_wrapper.cpp)
target_compile_definitions(uwot_shared PRIVATE UWOT_EXPORTS)

# Platform-specific library settings
if(WIN32)
    # Windows DLL settings
    set_target_properties(uwot_shared PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS TRUE
        OUTPUT_NAME "uwot"
    )
else()
    # Linux shared library settings
    set_target_properties(uwot_shared PROPERTIES
        OUTPUT_NAME "uwot"
        VERSION 1.0
        SOVERSION 1
    )
    # Link math library on Linux
    target_link_libraries(uwot_shared PRIVATE m)
endif()

# OpenMP support
if(OpenMP_CXX_FOUND)
    target_link_libraries(uwot_shared PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(uwot_shared PRIVATE UWOT_OPENMP)
    message(STATUS "OpenMP found and enabled")
else()
    message(STATUS "OpenMP not found - building without parallel support")
endif()

# Create test executable that links to the shared library
add_executable(uwot_test minimal_umap_test.cpp)
target_link_libraries(uwot_test PRIVATE uwot_shared)

# Create enhanced test with save/load functionality
add_executable(uwot_enhanced_test enhanced_test.cpp)
target_link_libraries(uwot_enhanced_test PRIVATE uwot_shared)
set(BUILD_ENHANCED_TEST ON)

# Platform-specific test executable settings
if(NOT WIN32)
    target_link_libraries(uwot_test PRIVATE m)
    target_link_libraries(uwot_enhanced_test PRIVATE m)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(uwot_test PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(uwot_enhanced_test PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(uwot_test PRIVATE UWOT_OPENMP)
    target_compile_definitions(uwot_enhanced_test PRIVATE UWOT_OPENMP)
endif()

# Installation rules for Linux
if(NOT WIN32)
    install(TARGETS uwot_shared
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(FILES uwot_simple_wrapper.h 
        DESTINATION include
    )
endif()

# Print build information
message(STATUS "")
message(STATUS "Build Configuration:")
message(STATUS "  System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
if(WIN32)
    message(STATUS "  Output: uwot.dll + uwot_test.exe + uwot_enhanced_test.exe")
else()
    message(STATUS "  Output: libuwot.so + uwot_test + uwot_enhanced_test")
endif()
message(STATUS "")