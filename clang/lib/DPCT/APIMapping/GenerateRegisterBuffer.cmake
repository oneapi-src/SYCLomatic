string(TIMESTAMP T0 "%s")
string(REGEX REPLACE ".*/(.*)$" "\\1" LIB_NAME ${LIB_PATH})
file(GLOB SRC_FILES ${LIB_PATH}/*.cu)
list(SUBLIST SRC_FILES ${BEGIN} ${STEP} SUB_FILES)
foreach(FILE ${SUB_FILES})
  string(REGEX REPLACE ".*/(.*).cu$" "\\1" API_NAME ${FILE})
  string(REPLACE "$" ":" QUERY_NAME ${API_NAME})
  execute_process(
    COMMAND ${CMAKE_BINARY_DIR}/../../../../bin/dpct-tmp --cuda-include-path=$ENV{CUDA_PATH}/include --query-api-mapping=${QUERY_NAME}
    OUTPUT_VARIABLE QUERY_STR
    ERROR_VARIABLE ERROR_STR)
  if("${ERROR_STR}" STREQUAL "")
    file(APPEND ${TARGET_FOLDER}APIMappingRegisterBuffer${LIB_NAME}${BEGIN}.def "REGIST(\"" "${API_NAME}" "\",\nR\"(" "${QUERY_STR}" ")\")\n")
  endif()
endforeach()
string(TIMESTAMP T1 "%s")
math(EXPR END "${BEGIN} + ${STEP} - 1")
math(EXPR T "${T1} - ${T0}")
message(STATUS "${LIB_NAME} API mapping ${BEGIN}~${END} generating time: ${T}s")
