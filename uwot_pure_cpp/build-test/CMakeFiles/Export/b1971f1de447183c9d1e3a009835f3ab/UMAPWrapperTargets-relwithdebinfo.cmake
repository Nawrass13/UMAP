#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "uwot_wrapper" for configuration "RelWithDebInfo"
set_property(TARGET uwot_wrapper APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(uwot_wrapper PROPERTIES
  IMPORTED_IMPLIB_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/uwot.lib"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/bin/uwot.dll"
  )

list(APPEND _cmake_import_check_targets uwot_wrapper )
list(APPEND _cmake_import_check_files_for_uwot_wrapper "${_IMPORT_PREFIX}/lib/uwot.lib" "${_IMPORT_PREFIX}/bin/uwot.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
