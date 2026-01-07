# FindMono.cmake
# Locates Mono installation
#
# sets:
# MONO_FOUND
# MONO_INCLUDE_DIRS
# MONO_LIBRARIES
# MONO_BIN_DIR (for DLLs)

if(WIN32)
    # Default locations on Windows
    set(MONO_SEARCH_PATHS
        "C:/Program Files/Mono"
        "C:/Program Files (x86)/Mono"
        "$ENV{MONO_HOME}"
    )
    
    find_path(MONO_INCLUDE_DIR
        NAMES mono/jit/jit.h
        PATHS ${MONO_SEARCH_PATHS}
        PATH_SUFFIXES include/mono-2.0
    )
    
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(MONO_LIB_NAME mono-2.0-sgen) # 64-bit
    else()
        set(MONO_LIB_NAME mono-2.0-sgen) # 32-bit (usually same name now)
    endif()

    find_library(MONO_LIBRARY
        NAMES ${MONO_LIB_NAME}
        PATHS ${MONO_SEARCH_PATHS}
        PATH_SUFFIXES lib
    )
    
    find_path(MONO_BIN_DIR
        NAMES ${MONO_LIB_NAME}.dll
        PATHS ${MONO_SEARCH_PATHS}
        PATH_SUFFIXES bin
    )

else()
    # Unix/Linux defaults
    find_package(PkgConfig QUIET)
    pkg_check_modules(MONO QUIET mono-2)
    
    if(MONO_FOUND)
        set(MONO_INCLUDE_DIR ${MONO_INCLUDE_DIRS})
        set(MONO_LIBRARY ${MONO_LIBRARIES})
    else()
        find_path(MONO_INCLUDE_DIR NAMES mono/jit/jit.h PATH_SUFFIXES mono-2.0)
        find_library(MONO_LIBRARY NAMES mono-2.0 mono-2.0-sgen)
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Mono DEFAULT_MSG MONO_LIBRARY MONO_INCLUDE_DIR)

if(MONO_FOUND)
    set(MONO_INCLUDE_DIRS ${MONO_INCLUDE_DIR})
    set(MONO_LIBRARIES ${MONO_LIBRARY})
endif()
