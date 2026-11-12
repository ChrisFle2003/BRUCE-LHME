if(NOT DEFINED REPORT_FILE)
  message(FATAL_ERROR "REPORT_FILE must name the canonical benchmark report")
endif()
if(NOT DEFINED RAW_FILE)
  message(FATAL_ERROR "RAW_FILE must name the canonical benchmark CSV")
endif()
if(NOT DEFINED VALIDATOR)
  message(FATAL_ERROR "VALIDATOR must name the canonical benchmark validator")
endif()

set(expected_raw_sha256 "d0dfeb0a3837a300c607f5340addf993cd3d832321dfa49c4d433b0bfa78fb7b")
file(SHA256 "${RAW_FILE}" actual_raw_sha256)
if(NOT actual_raw_sha256 STREQUAL expected_raw_sha256)
  message(FATAL_ERROR
    "Canonical benchmark CSV SHA-256 mismatch for ${RAW_FILE}: expected "
    "${expected_raw_sha256}, got ${actual_raw_sha256}")
endif()

file(READ "${REPORT_FILE}" report)
set(expected_report_hash_text "SHA-256\n`${expected_raw_sha256}`")
string(FIND "${report}" "${expected_report_hash_text}" report_hash_position)
if(report_hash_position EQUAL -1)
  message(FATAL_ERROR
    "Canonical benchmark report ${REPORT_FILE} does not state the expected SHA-256 "
    "for ${RAW_FILE}: ${expected_raw_sha256}")
endif()

execute_process(
  COMMAND "${VALIDATOR}"
  RESULT_VARIABLE validator_result
  OUTPUT_VARIABLE validator_stdout
  ERROR_VARIABLE validator_stderr
)
if(NOT validator_result EQUAL 0)
  message(FATAL_ERROR
    "Canonical benchmark CSV/report validation failed with exit code ${validator_result}.\n"
    "stdout:\n${validator_stdout}\n"
    "stderr:\n${validator_stderr}")
endif()
