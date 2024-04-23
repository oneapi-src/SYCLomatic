string(REPLACE "," ";" FILE_LIST ${FILE_LIST_STR})
foreach(FILE ${FILE_LIST})
  string(REGEX REPLACE "(.*).cu$" "\\1" API_NAME ${FILE})
  string(REPLACE "$" ":" QUERY_NAME ${API_NAME})
  execute_process(
    COMMAND ${CMAKE_BINARY_DIR}/../../../../bin/dpct-tmp --cuda-include-path=$ENV{CUDA_PATH}/include --query-api-mapping=${QUERY_NAME}
    OUTPUT_VARIABLE QUERY_STR
    ERROR_VARIABLE ERROR_STR)
  if("${ERROR_STR}" STREQUAL "")
    file(APPEND ${TARGET} "REGIST(\"" "${API_NAME}" "\",\nR\"(" "${QUERY_STR}" ")\")\n")
  endif()
endforeach()
