file(WRITE ${TARGET} "")
file(GLOB SRC_FILES ${SRC_DIRECTORY}*/*.cu)
message("generating " ${TARGET})
message("--cuda-include-path" $ENV{CUDA_PATH}/include)
message("CMAKE_BINARY_DIR " ${CMAKE_BINARY_DIR})
message("CMAKE_CURRENT_SOURCE_DIR " ${CMAKE_CURRENT_SOURCE_DIR})
foreach(FILE ${SRC_FILES})
  string(REGEX REPLACE ".*/(.*).cu$" "\\1" API_NAME ${FILE})
  string(REPLACE "$" ":" QUERY_NAME ${API_NAME})
  execute_process(
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../../../../bin/dpct-tmp --cuda-include-path=$ENV{CUDA_PATH}/include --query-api-mapping=${QUERY_NAME}
    OUTPUT_VARIABLE QUERY_STR
    ERROR_VARIABLE QUERY_STR
    OUTPUT_QUIET
    ERROR_QUIET)
  message(${CMAKE_CURRENT_SOURCE_DIR}/../../../../bin/dpct-tmp --cuda-include-path=$ENV{CUDA_PATH}/include --query-api-mapping=${QUERY_NAME})
  message("QUERY_STR: " ${QUERY_STR})
  file(APPEND ${TARGET} "registerEntry(\"" "${API_NAME}" "\",\nR\"(" "${QUERY_STR}" ")\");\n")
endforeach()
