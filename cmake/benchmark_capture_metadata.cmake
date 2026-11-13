cmake_minimum_required(VERSION 3.25)

# Create a versioned sidecar for an archived, non-canonical BRUCE LHME
# benchmark CSV. This script deliberately has no project-library dependency so
# it can be invoked with `cmake -P` after a configured benchmark run.

foreach(required CAPTURE_CSV CAPTURE_OUTPUT)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

if(DEFINED CAPTURE_CLASS AND NOT CAPTURE_CLASS STREQUAL "non-canonical")
  message(FATAL_ERROR
    "CAPTURE_CLASS must be 'non-canonical'; this tool must not describe canonical measurements")
endif()

if(NOT EXISTS "${CAPTURE_CSV}" OR IS_DIRECTORY "${CAPTURE_CSV}")
  message(FATAL_ERROR "CAPTURE_CSV must name an existing CSV file: ${CAPTURE_CSV}")
endif()
if(NOT CAPTURE_CSV MATCHES "\\.csv$")
  message(FATAL_ERROR "CAPTURE_CSV must use the .csv extension: ${CAPTURE_CSV}")
endif()
if(EXISTS "${CAPTURE_OUTPUT}")
  message(FATAL_ERROR "Refusing to overwrite existing capture metadata: ${CAPTURE_OUTPUT}")
endif()
if(NOT CAPTURE_OUTPUT MATCHES "\\.json$")
  message(FATAL_ERROR "CAPTURE_OUTPUT must use the .json extension: ${CAPTURE_OUTPUT}")
endif()
get_filename_component(capture_output_directory "${CAPTURE_OUTPUT}" DIRECTORY)
if(NOT IS_DIRECTORY "${capture_output_directory}")
  message(FATAL_ERROR
    "CAPTURE_OUTPUT parent directory must already exist: ${capture_output_directory}")
endif()

file(REAL_PATH "${CAPTURE_CSV}" capture_csv_real)
file(REAL_PATH
  "${CMAKE_CURRENT_LIST_DIR}/../docs/benchmark-raw-2026-11-11-v3.csv"
  canonical_csv_real)
if(capture_csv_real STREQUAL canonical_csv_real)
  message(FATAL_ERROR
    "Refusing to generate non-canonical metadata for the canonical -v3 CSV: ${CAPTURE_CSV}")
endif()
file(SHA256 "${capture_csv_real}" capture_csv_sha256)

function(json_escape input output)
  set(value "${input}")
  string(REPLACE "\\" "\\\\" value "${value}")
  string(REPLACE "\"" "\\\"" value "${value}")
  string(REPLACE "\n" "\\n" value "${value}")
  string(REPLACE "\r" "\\r" value "${value}")
  string(REPLACE "\t" "\\t" value "${value}")
  set(${output} "${value}" PARENT_SCOPE)
endfunction()

function(json_string value output)
  json_escape("${value}" escaped)
  set(${output} "\"${escaped}\"" PARENT_SCOPE)
endfunction()

function(json_array_from_list output)
  set(result "[")
  set(first TRUE)
  foreach(item IN LISTS ARGN)
    json_string("${item}" encoded)
    if(first)
      set(first FALSE)
    else()
      string(APPEND result ", ")
    endif()
    string(APPEND result "${encoded}")
  endforeach()
  string(APPEND result "]")
  set(${output} "${result}" PARENT_SCOPE)
endfunction()

function(validate_json_array label json_value output)
  string(JSON json_type ERROR_VARIABLE json_error TYPE "${json_value}")
  if(NOT json_error STREQUAL "NOTFOUND")
    message(FATAL_ERROR "${label} must be valid JSON: ${json_error}")
  endif()
  if(NOT json_type STREQUAL "ARRAY")
    message(FATAL_ERROR "${label} must be a JSON array, not ${json_type}")
  endif()
  set(${output} "${json_value}" PARENT_SCOPE)
endfunction()

function(cache_value cache_file variable fallback output)
  set(value "${fallback}")
  if(EXISTS "${cache_file}")
    file(STRINGS "${cache_file}" matching_lines REGEX "^${variable}:[^=]*=")
    if(matching_lines)
      list(GET matching_lines 0 matching_line)
      string(REGEX REPLACE "^[^=]*=" "" value "${matching_line}")
    endif()
  endif()
  set(${output} "${value}" PARENT_SCOPE)
endfunction()

function(configured_compiler_value cache_file variable fallback output)
  set(value "${fallback}")
  if(EXISTS "${cache_file}")
    get_filename_component(cache_directory "${cache_file}" DIRECTORY)
    file(GLOB compiler_configuration_files
      "${cache_directory}/CMakeFiles/*/CMakeCXXCompiler.cmake")
    list(SORT compiler_configuration_files)
    foreach(compiler_configuration_file IN LISTS compiler_configuration_files)
      file(STRINGS "${compiler_configuration_file}" matching_lines
        REGEX "^set\\(${variable} \"")
      if(matching_lines)
        list(GET matching_lines 0 matching_line)
        string(REGEX REPLACE
          "^set\\(${variable} \"([^\"]*)\"\\)$"
          "\\1"
          value
          "${matching_line}")
        break()
      endif()
    endforeach()
  endif()
  set(${output} "${value}" PARENT_SCOPE)
endfunction()

function(value_from_override override_variable detected_value output)
  if(DEFINED ${override_variable} AND NOT "${${override_variable}}" STREQUAL "")
    set(value "${${override_variable}}")
  elseif(NOT "${detected_value}" STREQUAL "")
    set(value "${detected_value}")
  else()
    set(value "not available")
  endif()
  set(${output} "${value}" PARENT_SCOPE)
endfunction()

set(capture_cache_file "")
if(DEFINED CMAKE_CACHE_FILE AND NOT "${CMAKE_CACHE_FILE}" STREQUAL "")
  if(NOT EXISTS "${CMAKE_CACHE_FILE}")
    message(FATAL_ERROR "CMAKE_CACHE_FILE does not exist: ${CMAKE_CACHE_FILE}")
  endif()
  set(capture_cache_file "${CMAKE_CACHE_FILE}")
endif()

cache_value("${capture_cache_file}" "CMAKE_BUILD_TYPE" "" detected_build_configuration)
cache_value("${capture_cache_file}" "CMAKE_GENERATOR" "" detected_generator)
cache_value("${capture_cache_file}" "CMAKE_CXX_COMPILER" "" detected_compiler_path)
configured_compiler_value("${capture_cache_file}" "CMAKE_CXX_COMPILER_ID" "" detected_compiler_id)
configured_compiler_value(
  "${capture_cache_file}" "CMAKE_CXX_COMPILER_VERSION" "" detected_compiler_version)
value_from_override(BUILD_CONFIGURATION "${detected_build_configuration}" build_configuration)
value_from_override(GENERATOR "${detected_generator}" generator)
value_from_override(COMPILER_ID "${detected_compiler_id}" compiler_id)
value_from_override(COMPILER_PATH "${detected_compiler_path}" compiler_path)
value_from_override(COMPILER_VERSION "${detected_compiler_version}" compiler_version)
value_from_override(CMAKE_VERSION_OVERRIDE "${CMAKE_VERSION}" cmake_version)

if(DEFINED CMAKE_OPTIONS_JSON)
  validate_json_array("CMAKE_OPTIONS_JSON" "${CMAKE_OPTIONS_JSON}" cmake_options_json)
elseif(capture_cache_file)
  set(cmake_options)
  foreach(option
      BUILD_TESTING
      BRUCE_BUILD_BENCHMARKS
      BRUCE_BUILD_TOOLS
      BRUCE_BUILD_EXAMPLES
      BRUCE_SANITIZER)
    cache_value("${capture_cache_file}" "${option}" "not available" option_value)
    list(APPEND cmake_options "${option}=${option_value}")
  endforeach()
  json_array_from_list(cmake_options_json ${cmake_options})
else()
  set(cmake_options_json "[\"not available\"]")
endif()

if(DEFINED CAPTURED_AT_UTC AND NOT "${CAPTURED_AT_UTC}" STREQUAL "")
  set(captured_at_utc "${CAPTURED_AT_UTC}")
else()
  string(TIMESTAMP captured_at_utc "%Y-%m-%dT%H:%M:%SZ" UTC)
endif()
if(NOT captured_at_utc MATCHES
    "^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z$")
  message(FATAL_ERROR
    "CAPTURED_AT_UTC must use UTC RFC 3339 second precision, for example 2026-11-11T00:00:00Z")
endif()

set(detected_operating_system "${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION}")
set(detected_platform_label "${CMAKE_HOST_SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_PROCESSOR}")
set(detected_cpu_model "")
set(detected_logical_cpu_count "")
set(detected_physical_memory_bytes "")
set(detected_cpu_affinity "")
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  if(EXISTS "/proc/cpuinfo")
    file(READ "/proc/cpuinfo" cpuinfo_contents)
    string(REGEX MATCH "model name[ \t]*:[ \t]*([^\r\n]+)" cpu_model_match "${cpuinfo_contents}")
    if(cpu_model_match)
      set(detected_cpu_model "${CMAKE_MATCH_1}")
    endif()
  endif()
  if(EXISTS "/proc/meminfo")
    file(READ "/proc/meminfo" meminfo_contents)
    string(REGEX MATCH "MemTotal:[ \t]*([0-9]+)[ \t]*kB" memory_match "${meminfo_contents}")
    if(memory_match)
      math(EXPR detected_physical_memory_bytes "${CMAKE_MATCH_1} * 1024")
    endif()
  endif()
  if(EXISTS "/proc/self/status")
    file(READ "/proc/self/status" process_status_contents)
    string(REGEX MATCH "Cpus_allowed_list:[ \t]*([^\r\n]+)" affinity_match "${process_status_contents}")
    if(affinity_match)
      string(STRIP "${CMAKE_MATCH_1}" detected_cpu_affinity)
    endif()
  endif()
endif()
cmake_host_system_information(RESULT detected_logical_cpu_count_query QUERY NUMBER_OF_LOGICAL_CORES)
if(NOT "${detected_logical_cpu_count_query}" STREQUAL "")
  set(detected_logical_cpu_count "${detected_logical_cpu_count_query}")
endif()
value_from_override(OPERATING_SYSTEM "${detected_operating_system}" operating_system)
value_from_override(PLATFORM_LABEL "${detected_platform_label}" platform_label)
value_from_override(CPU_MODEL "${detected_cpu_model}" cpu_model)
value_from_override(LOGICAL_CPU_COUNT "${detected_logical_cpu_count}" logical_cpu_count)
value_from_override(PHYSICAL_MEMORY_BYTES "${detected_physical_memory_bytes}" physical_memory_bytes)
value_from_override(CPU_AFFINITY "${detected_cpu_affinity}" cpu_affinity)

value_from_override(ITERATION_PROTOCOL "" iteration_protocol)
value_from_override(WARMUP_PROTOCOL "" warmup_protocol)
if(DEFINED PLATFORM_LIMITS_JSON)
  validate_json_array("PLATFORM_LIMITS_JSON" "${PLATFORM_LIMITS_JSON}" platform_limits_json)
else()
  set(platform_limits_json "[\"not available\"]")
endif()

value_from_override(REVISION_IDENTIFIER "" revision_identifier)
if(revision_identifier STREQUAL "not available")
  set(revision_availability "not available")
else()
  set(revision_availability "provided")
endif()

foreach(value_name
    captured_at_utc
    capture_csv_sha256
    build_configuration
    compiler_id
    compiler_path
    compiler_version
    cmake_version
    generator
    operating_system
    platform_label
    cpu_model
    logical_cpu_count
    physical_memory_bytes
    cpu_affinity
    iteration_protocol
    warmup_protocol
    revision_identifier
    revision_availability)
  json_string("${${value_name}}" "json_${value_name}")
endforeach()
json_string("${CAPTURE_CSV}" json_capture_csv_path)

set(metadata "{\n")
string(APPEND metadata "  \"schema\": \"bruce.lhme.benchmark-capture\",\n")
string(APPEND metadata "  \"schema_version\": 1,\n")
string(APPEND metadata "  \"capture_class\": \"non-canonical\",\n")
string(APPEND metadata "  \"captured_at_utc\": ${json_captured_at_utc},\n")
string(APPEND metadata "  \"csv\": {\n")
string(APPEND metadata "    \"path\": ${json_capture_csv_path},\n")
string(APPEND metadata "    \"sha256\": ${json_capture_csv_sha256}\n")
string(APPEND metadata "  },\n")
string(APPEND metadata "  \"revision\": {\n")
string(APPEND metadata "    \"identifier\": ${json_revision_identifier},\n")
string(APPEND metadata "    \"availability\": ${json_revision_availability}\n")
string(APPEND metadata "  },\n")
string(APPEND metadata "  \"build\": {\n")
string(APPEND metadata "    \"configuration\": ${json_build_configuration},\n")
string(APPEND metadata "    \"cmake_options\": ${cmake_options_json},\n")
string(APPEND metadata "    \"compiler\": {\n")
string(APPEND metadata "      \"id\": ${json_compiler_id},\n")
string(APPEND metadata "      \"path\": ${json_compiler_path},\n")
string(APPEND metadata "      \"version\": ${json_compiler_version}\n")
string(APPEND metadata "    },\n")
string(APPEND metadata "    \"cmake_version\": ${json_cmake_version},\n")
string(APPEND metadata "    \"generator\": ${json_generator}\n")
string(APPEND metadata "  },\n")
string(APPEND metadata "  \"environment\": {\n")
string(APPEND metadata "    \"platform_label\": ${json_platform_label},\n")
string(APPEND metadata "    \"operating_system\": ${json_operating_system},\n")
string(APPEND metadata "    \"cpu_model\": ${json_cpu_model},\n")
string(APPEND metadata "    \"logical_cpu_count\": ${json_logical_cpu_count},\n")
string(APPEND metadata "    \"physical_memory_bytes\": ${json_physical_memory_bytes},\n")
string(APPEND metadata "    \"cpu_affinity\": ${json_cpu_affinity}\n")
string(APPEND metadata "  },\n")
string(APPEND metadata "  \"protocol\": {\n")
string(APPEND metadata "    \"iterations\": ${json_iteration_protocol},\n")
string(APPEND metadata "    \"warmup\": ${json_warmup_protocol}\n")
string(APPEND metadata "  },\n")
string(APPEND metadata "  \"platform_limits\": ${platform_limits_json}\n")
string(APPEND metadata "}\n")

file(WRITE "${CAPTURE_OUTPUT}" "${metadata}")
