if(NOT DEFINED BRUCE_BUILD_DIR OR NOT DEFINED BRUCE_STAGE_DIR OR
   NOT DEFINED BRUCE_INSTALL_LIBDIR)
  message(FATAL_ERROR "package-layout test requires build, stage, and library directories")
endif()

file(REMOVE_RECURSE "${BRUCE_STAGE_DIR}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${BRUCE_BUILD_DIR}" --prefix "${BRUCE_STAGE_DIR}"
  RESULT_VARIABLE install_result
  OUTPUT_VARIABLE install_stdout
  ERROR_VARIABLE install_stderr)

if(NOT install_result EQUAL 0)
  message(FATAL_ERROR
    "staged install failed with code ${install_result}\nstdout:\n${install_stdout}\nstderr:\n${install_stderr}")
endif()

set(package_directory
    "${BRUCE_STAGE_DIR}/${BRUCE_INSTALL_LIBDIR}/cmake/bruce_lhme")
foreach(package_file IN ITEMS
    bruce_lhmeConfig.cmake
    bruce_lhmeConfigVersion.cmake
    bruce_lhmeTargets.cmake)
  if(NOT EXISTS "${package_directory}/${package_file}")
    message(FATAL_ERROR
      "missing installed package file: ${package_directory}/${package_file}\ninstall stdout:\n${install_stdout}\ninstall stderr:\n${install_stderr}")
  endif()
endforeach()

# Pre-1.0 minor releases are distinct public C++ API/ABI generations.  Run
# real Config-mode consumer projects because the package config resolves the
# Threads dependency, which requires an enabled language.
set(version_probe_source "${BRUCE_STAGE_DIR}/version_probe_source")
set(version_probe_build_root "${BRUCE_STAGE_DIR}/version_probe_build")
file(REMOVE_RECURSE "${version_probe_source}" "${version_probe_build_root}")
file(MAKE_DIRECTORY "${version_probe_source}")

function(configure_version_probe requested_version expect_success)
  file(WRITE "${version_probe_source}/CMakeLists.txt"
    "cmake_minimum_required(VERSION 3.25)\n"
    "project(bruce_lhme_version_probe LANGUAGES CXX)\n"
    "find_package(bruce_lhme ${requested_version} CONFIG REQUIRED)\n"
    "if(NOT TARGET bruce_lhme::core)\n"
    "  message(FATAL_ERROR \\\"bruce_lhme::core was not exported\\\")\n"
    "endif()\n")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${version_probe_source}"
      -B "${version_probe_build_root}/${requested_version}"
      "-DCMAKE_PREFIX_PATH=${BRUCE_STAGE_DIR}"
    RESULT_VARIABLE probe_result
    OUTPUT_VARIABLE probe_stdout
    ERROR_VARIABLE probe_stderr)
  if(expect_success AND NOT probe_result EQUAL 0)
    message(FATAL_ERROR
      "the staged package rejected bruce_lhme ${requested_version}\nstdout:\n${probe_stdout}\nstderr:\n${probe_stderr}")
  endif()
  if(NOT expect_success AND probe_result EQUAL 0)
    message(FATAL_ERROR
      "the staged package unexpectedly accepts bruce_lhme ${requested_version}")
  endif()
endfunction()

configure_version_probe(0.1 FALSE)
configure_version_probe(0.2 TRUE)
