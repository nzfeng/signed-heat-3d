cmake_minimum_required(VERSION 3.5.0)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Maybe stop from CMAKEing in the wrong place
if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Source and build directories cannot be the same. Go use the /build directory.")
endif()

# List of .cpp files to be compiled
SET(SRCS
  signed_heat_3d.cpp
  signed_heat_grid_solver.cpp
  signed_heat_tet_solver.cpp  
)

SET(INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../include/signedheat3d/")
SET(HEADERS
  ${INCLUDE_ROOT}/signed_heat_3d.h
  ${INCLUDE_ROOT}/signed_heat_grid_solver.h
  ${INCLUDE_ROOT}/signed_heat_tet_solver.h
)

# Create a single library for the project
add_library(signed-heat-3d ${SRCS} ${HEADERS})

option(SHM_NO_AMGCL "Do not use AMGCL" OFF)
if (SHM_NO_AMGCL)
  add_definitions(-DSHM_NO_AMGCL=)
  message(STATUS "Not using AMGCL")
else()
  add_definitions(-USHM_NO_AMGCL)
  message(STATUS "Using AMGCL")
endif()

add_subdirectory(../deps/geometry-central deps/geometry-central)
add_subdirectory(../deps/tetgen deps/tetgen)
add_subdirectory(../deps/amgcl deps/amgcl)

# Includes from this project
target_include_directories(signed-heat-3d PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")

# Add all includes and link libraries from dependencies
target_include_directories(signed-heat-3d PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../deps/tetgen/")
target_include_directories(signed-heat-3d PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../deps/libigl/include")
target_link_libraries(signed-heat-3d geometry-central tetgen amgcl::amgcl)

# Set compiler properties for the library
target_compile_features(signed-heat-3d PUBLIC cxx_std_11)
set_target_properties(signed-heat-3d PROPERTIES
  CXX_STANDARD_REQUIRED TRUE
  CXX_EXTENSIONS OFF
)
target_compile_definitions(signed-heat-3d PUBLIC NOMINMAX _USE_MATH_DEFINES)

# Export symbols if DLL is requested
if(MSVC AND BUILD_SHARED_LIBS)
  set_target_properties(signed-heat-3d PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
