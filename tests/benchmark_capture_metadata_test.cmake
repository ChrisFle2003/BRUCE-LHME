cmake_minimum_required(VERSION 3.25)

foreach(required TEST_OUTPUT_ROOT CAPTURE_SCRIPT)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

set(owner_marker ".bruce-lhme-benchmark-capture-metadata-owned")
set(owner_contents "BRUCE LHME benchmark capture metadata CTest artifact\n")
set(marker_path "${TEST_OUTPUT_ROOT}/${owner_marker}")
if(EXISTS "${TEST_OUTPUT_ROOT}")
  if(NOT EXISTS "${marker_path}")
    message(FATAL_ERROR
      "Refusing to remove unmanaged benchmark capture metadata directory: ${TEST_OUTPUT_ROOT}")
  endif()
  file(READ "${marker_path}" existing_owner)
  if(NOT existing_owner STREQUAL owner_contents)
    message(FATAL_ERROR
      "Refusing to remove benchmark capture metadata directory with an unexpected ownership marker: ${TEST_OUTPUT_ROOT}")
  endif()
  file(REMOVE_RECURSE "${TEST_OUTPUT_ROOT}")
endif()
file(MAKE_DIRECTORY "${TEST_OUTPUT_ROOT}")
file(WRITE "${marker_path}" "${owner_contents}")

set(csv_path "${TEST_OUTPUT_ROOT}/future-noncanonical.csv")
set(metadata_path "${TEST_OUTPUT_ROOT}/future-noncanonical.capture.json")
file(WRITE "${csv_path}"
  "benchmark,case_id,metric,iteration,unit,value,validation,notes\n"
  "\"fixture\",\"case\",\"elapsed\",0,\"ns/op\",1.000000000,\"confirmed\",\"deterministic fixture\"\n")
file(SHA256 "${csv_path}" expected_sha256)

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DCAPTURE_CSV=${csv_path}"
    "-DCAPTURE_OUTPUT=${metadata_path}"
    "-DCAPTURED_AT_UTC=2026-11-11T00:00:00Z"
    "-DBUILD_CONFIGURATION=Release"
    "-DCMAKE_OPTIONS_JSON=[\"BUILD_TESTING=ON\",\"BRUCE_BUILD_BENCHMARKS=ON\"]"
    "-DCOMPILER_ID=GNU"
    "-DCOMPILER_PATH=/usr/bin/g++"
    "-DCOMPILER_VERSION=14.2.0"
    "-DCMAKE_VERSION_OVERRIDE=3.31.6"
    "-DGENERATOR=Ninja"
    "-DOPERATING_SYSTEM=TestOS 1.0"
    "-DPLATFORM_LABEL=test-platform"
    "-DCPU_MODEL=deterministic CPU"
    "-DLOGICAL_CPU_COUNT=2"
    "-DPHYSICAL_MEMORY_BYTES=8589934592"
    "-DCPU_AFFINITY=0-1"
    "-DITERATION_PROTOCOL=11 timed iterations per case"
    "-DWARMUP_PROTOCOL=1 unscored warm-up iteration per case"
    "-DPLATFORM_LIMITS_JSON=[\"fixture platform limit\"]"
    "-DREVISION_IDENTIFIER=not available"
    -P "${CAPTURE_SCRIPT}"
  RESULT_VARIABLE capture_result
  OUTPUT_VARIABLE capture_stdout
  ERROR_VARIABLE capture_stderr
)
if(NOT capture_result EQUAL 0)
  message(FATAL_ERROR
    "Benchmark capture metadata generation failed with exit code ${capture_result}.\n"
    "stdout:\n${capture_stdout}\n"
    "stderr:\n${capture_stderr}")
endif()

if(NOT EXISTS "${metadata_path}")
  message(FATAL_ERROR "Benchmark capture metadata file was not created: ${metadata_path}")
endif()
file(READ "${metadata_path}" metadata_json)

function(require_json_value label expected)
  string(JSON actual GET "${metadata_json}" ${ARGN})
  if(NOT actual STREQUAL expected)
    message(FATAL_ERROR
      "${label} is '${actual}'; expected '${expected}'")
  endif()
endfunction()

function(require_json_type label expected)
  string(JSON actual_type TYPE "${metadata_json}" ${ARGN})
  if(NOT actual_type STREQUAL expected)
    message(FATAL_ERROR
      "${label} has JSON type '${actual_type}'; expected '${expected}'")
  endif()
endfunction()

require_json_value("schema" "bruce.lhme.benchmark-capture" schema)
require_json_value("schema version" "1" schema_version)
require_json_value("capture class" "non-canonical" capture_class)
require_json_value("capture date" "2026-11-11T00:00:00Z" captured_at_utc)
require_json_value("CSV path" "${csv_path}" csv path)
require_json_value("CSV SHA-256" "${expected_sha256}" csv sha256)
require_json_value("revision identifier" "not available" revision identifier)
require_json_value("revision availability" "not available" revision availability)
require_json_value("platform label" "test-platform" environment platform_label)
require_json_value("operating system" "TestOS 1.0" environment operating_system)
require_json_value("CPU model" "deterministic CPU" environment cpu_model)
require_json_value("logical CPU count" "2" environment logical_cpu_count)
require_json_value("physical memory bytes" "8589934592" environment physical_memory_bytes)
require_json_value("CPU affinity" "0-1" environment cpu_affinity)
require_json_value("build configuration" "Release" build configuration)
require_json_value("compiler ID" "GNU" build compiler id)
require_json_value("compiler path" "/usr/bin/g++" build compiler path)
require_json_value("compiler version" "14.2.0" build compiler version)
require_json_value("CMake version" "3.31.6" build cmake_version)
require_json_value("generator" "Ninja" build generator)
require_json_value("iteration protocol" "11 timed iterations per case" protocol iterations)
require_json_value("warm-up protocol" "1 unscored warm-up iteration per case" protocol warmup)
require_json_type("CMake options" ARRAY build cmake_options)
require_json_type("platform limits" ARRAY platform_limits)
require_json_value("first CMake option" "BUILD_TESTING=ON" build cmake_options 0)
require_json_value("second CMake option" "BRUCE_BUILD_BENCHMARKS=ON" build cmake_options 1)
require_json_value("first platform limit" "fixture platform limit" platform_limits 0)

file(SHA256 "${csv_path}" post_generation_sha256)
if(NOT post_generation_sha256 STREQUAL expected_sha256)
  message(FATAL_ERROR
    "The fixture CSV changed while generating metadata; refusing an unsafe SHA-256 reference")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DCAPTURE_CSV=${csv_path}"
    "-DCAPTURE_OUTPUT=${metadata_path}"
    "-DREVISION_IDENTIFIER=not available"
    -P "${CAPTURE_SCRIPT}"
  RESULT_VARIABLE overwrite_result
  OUTPUT_VARIABLE overwrite_stdout
  ERROR_VARIABLE overwrite_stderr
)
if(overwrite_result EQUAL 0)
  message(FATAL_ERROR
    "Benchmark capture metadata generation unexpectedly overwrote an existing output file")
endif()
file(READ "${metadata_path}" metadata_after_overwrite_attempt)
if(NOT metadata_after_overwrite_attempt STREQUAL metadata_json)
  message(FATAL_ERROR
    "A rejected metadata overwrite attempt changed the existing metadata file")
endif()

set(configured_build_directory "${TEST_OUTPUT_ROOT}/configured-build")
set(configured_cache "${configured_build_directory}/CMakeCache.txt")
set(configured_metadata "${TEST_OUTPUT_ROOT}/cache-derived.capture.json")
file(MAKE_DIRECTORY "${configured_build_directory}/CMakeFiles/3.25.0")
file(WRITE "${configured_cache}"
  "CMAKE_BUILD_TYPE:STRING=RelWithDebInfo\n"
  "CMAKE_GENERATOR:INTERNAL=Fixture Generator\n"
  "CMAKE_CXX_COMPILER:FILEPATH=/fixture/bin/c++\n"
  "BUILD_TESTING:BOOL=ON\n"
  "BRUCE_BUILD_BENCHMARKS:BOOL=ON\n"
  "BRUCE_BUILD_TOOLS:BOOL=OFF\n"
  "BRUCE_BUILD_EXAMPLES:BOOL=OFF\n"
  "BRUCE_SANITIZER:STRING=none\n")
file(WRITE "${configured_build_directory}/CMakeFiles/3.25.0/CMakeCXXCompiler.cmake"
  "set(CMAKE_CXX_COMPILER_ID \"FixtureCXX\")\n"
  "set(CMAKE_CXX_COMPILER_VERSION \"1.2.3\")\n")

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DCAPTURE_CSV=${csv_path}"
    "-DCAPTURE_OUTPUT=${configured_metadata}"
    "-DCAPTURED_AT_UTC=2026-11-11T00:00:01Z"
    "-DCMAKE_CACHE_FILE=${configured_cache}"
    "-DITERATION_PROTOCOL=cache-derived protocol"
    "-DWARMUP_PROTOCOL=cache-derived warm-up"
    "-DREVISION_IDENTIFIER=not available"
    -P "${CAPTURE_SCRIPT}"
  RESULT_VARIABLE configured_capture_result
  OUTPUT_VARIABLE configured_capture_stdout
  ERROR_VARIABLE configured_capture_stderr
)
if(NOT configured_capture_result EQUAL 0)
  message(FATAL_ERROR
    "Cache-derived benchmark capture metadata generation failed with exit code ${configured_capture_result}.\n"
    "stdout:\n${configured_capture_stdout}\n"
    "stderr:\n${configured_capture_stderr}")
endif()
file(READ "${configured_metadata}" configured_metadata_json)

function(require_configured_json_value label expected)
  string(JSON actual GET "${configured_metadata_json}" ${ARGN})
  if(NOT actual STREQUAL expected)
    message(FATAL_ERROR
      "Cache-derived ${label} is '${actual}'; expected '${expected}'")
  endif()
endfunction()

require_configured_json_value("build configuration" "RelWithDebInfo" build configuration)
require_configured_json_value("generator" "Fixture Generator" build generator)
require_configured_json_value("compiler ID" "FixtureCXX" build compiler id)
require_configured_json_value("compiler path" "/fixture/bin/c++" build compiler path)
require_configured_json_value("compiler version" "1.2.3" build compiler version)
require_configured_json_value("first CMake option" "BUILD_TESTING=ON" build cmake_options 0)
require_configured_json_value(
  "second CMake option" "BRUCE_BUILD_BENCHMARKS=ON" build cmake_options 1)
require_configured_json_value("third CMake option" "BRUCE_BUILD_TOOLS=OFF" build cmake_options 2)
require_configured_json_value(
  "fourth CMake option" "BRUCE_BUILD_EXAMPLES=OFF" build cmake_options 3)
require_configured_json_value("fifth CMake option" "BRUCE_SANITIZER=none" build cmake_options 4)

get_filename_component(repository_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(canonical_csv "${repository_root}/docs/benchmark-raw-2026-11-11-v3.csv")
set(rejected_canonical_metadata "${TEST_OUTPUT_ROOT}/canonical.capture.json")
file(SHA256 "${canonical_csv}" canonical_sha256_before)
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DCAPTURE_CSV=${canonical_csv}"
    "-DCAPTURE_OUTPUT=${rejected_canonical_metadata}"
    "-DREVISION_IDENTIFIER=not available"
    -P "${CAPTURE_SCRIPT}"
  RESULT_VARIABLE canonical_capture_result
  OUTPUT_VARIABLE canonical_capture_stdout
  ERROR_VARIABLE canonical_capture_stderr
)
if(canonical_capture_result EQUAL 0)
  message(FATAL_ERROR
    "Benchmark capture metadata generation unexpectedly accepted the canonical -v3 CSV")
endif()
if(EXISTS "${rejected_canonical_metadata}")
  message(FATAL_ERROR
    "A rejected canonical capture created metadata: ${rejected_canonical_metadata}")
endif()
file(SHA256 "${canonical_csv}" canonical_sha256_after)
if(NOT canonical_sha256_after STREQUAL canonical_sha256_before)
  message(FATAL_ERROR
    "The rejected canonical capture changed the canonical -v3 CSV")
endif()

file(REMOVE_RECURSE "${TEST_OUTPUT_ROOT}")
