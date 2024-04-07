ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_BINARY_DIR}/bin/codepin-report
                COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_SOURCE_DIR}/codepin-report
                        ${CMAKE_BINARY_DIR}/bin)
list(APPEND dpct-codepin-report-dependends ${CMAKE_BINARY_DIR}/bin/codepin-report)

add_custom_target(dpct-codepin-report-files ALL DEPENDS ${dpct-codepin-report-dependends})

install(PROGRAMS codepin-report
    DESTINATION ./bin
    COMPONENT dpct-codepin-report)

add_llvm_install_targets(install-dpct-codepin-report
                        DEPENDS dpct-codepin-report-files
                        COMPONENT dpct-codepin-report)
