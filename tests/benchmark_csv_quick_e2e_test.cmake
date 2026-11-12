if(NOT DEFINED TEST_OUTPUT_ROOT OR NOT DEFINED VALIDATOR)
  message(FATAL_ERROR "TEST_OUTPUT_ROOT and VALIDATOR are required")
endif()
foreach(required FAIR_EXECUTABLE KEVIN_EXECUTABLE HYBRID_EXECUTABLE TABLE_EXECUTABLE STRESS_EXECUTABLE)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

set(owner_marker ".bruce-lhme-benchmark-csv-e2e-owned")
set(owner_contents "BRUCE LHME benchmark CSV CTest artifact\n")
set(marker_path "${TEST_OUTPUT_ROOT}/${owner_marker}")
if(EXISTS "${TEST_OUTPUT_ROOT}")
  if(NOT EXISTS "${marker_path}")
    message(FATAL_ERROR
      "Refusing to remove unmanaged benchmark CSV test directory: ${TEST_OUTPUT_ROOT}")
  endif()
  file(READ "${marker_path}" existing_owner)
  if(NOT existing_owner STREQUAL owner_contents)
    message(FATAL_ERROR
      "Refusing to remove benchmark CSV test directory with an unexpected ownership marker: ${TEST_OUTPUT_ROOT}")
  endif()
  file(REMOVE_RECURSE "${TEST_OUTPUT_ROOT}")
endif()
file(MAKE_DIRECTORY "${TEST_OUTPUT_ROOT}")
file(WRITE "${marker_path}" "${owner_contents}")

function(run_quick benchmark executable)
  if(NOT EXISTS "${executable}")
    message(FATAL_ERROR "Benchmark executable is missing: ${executable}")
  endif()
  execute_process(
    COMMAND "${executable}" --quick --csv "${TEST_OUTPUT_ROOT}/${benchmark}.csv"
    RESULT_VARIABLE result
    OUTPUT_FILE "${TEST_OUTPUT_ROOT}/${benchmark}.stdout"
    ERROR_FILE "${TEST_OUTPUT_ROOT}/${benchmark}.stderr"
  )
  if(NOT result EQUAL 0)
    file(READ "${TEST_OUTPUT_ROOT}/${benchmark}.stdout" captured_stdout)
    file(READ "${TEST_OUTPUT_ROOT}/${benchmark}.stderr" captured_stderr)
    message(FATAL_ERROR
      "${benchmark} --quick --csv failed with exit code ${result}.\n"
      "stdout:\n${captured_stdout}\n"
      "stderr:\n${captured_stderr}")
  endif()
endfunction()

run_quick(kevin "${KEVIN_EXECUTABLE}")
run_quick(hybrid "${HYBRID_EXECUTABLE}")

set(validator_arguments
  --kevin-csv "${TEST_OUTPUT_ROOT}/kevin.csv"
  --kevin-stdout "${TEST_OUTPUT_ROOT}/kevin.stdout"
  --hybrid-csv "${TEST_OUTPUT_ROOT}/hybrid.csv"
  --hybrid-stdout "${TEST_OUTPUT_ROOT}/hybrid.stdout"
)
if(LINUX_SUPPORTED)
  run_quick(fair "${FAIR_EXECUTABLE}")
  run_quick(table "${TABLE_EXECUTABLE}")
  run_quick(stress "${STRESS_EXECUTABLE}")
  list(APPEND validator_arguments
    --fair-csv "${TEST_OUTPUT_ROOT}/fair.csv"
    --fair-stdout "${TEST_OUTPUT_ROOT}/fair.stdout"
    --table-csv "${TEST_OUTPUT_ROOT}/table.csv"
    --table-stdout "${TEST_OUTPUT_ROOT}/table.stdout"
    --stress-csv "${TEST_OUTPUT_ROOT}/stress.csv"
    --stress-stdout "${TEST_OUTPUT_ROOT}/stress.stdout"
  )
else()
  foreach(skipped_benchmark fair table stress)
    foreach(extension csv stdout stderr)
      if(EXISTS "${TEST_OUTPUT_ROOT}/${skipped_benchmark}.${extension}")
        message(FATAL_ERROR
          "${skipped_benchmark}.${extension} exists although the non-Linux path must not execute it")
      endif()
    endforeach()
  endforeach()
  message(STATUS
    "SKIP fair/table/stress benchmark CSV end-to-end paths: isolated process/RSS measurement is Linux-only")
endif()

execute_process(
  COMMAND "${VALIDATOR}" ${validator_arguments}
  RESULT_VARIABLE validator_result
  OUTPUT_VARIABLE validator_stdout
  ERROR_VARIABLE validator_stderr
)
if(NOT validator_result EQUAL 0)
  message(FATAL_ERROR
    "Benchmark CSV end-to-end validation failed with exit code ${validator_result}.\n"
    "stdout:\n${validator_stdout}\n"
    "stderr:\n${validator_stderr}")
endif()

file(REMOVE_RECURSE "${TEST_OUTPUT_ROOT}")
