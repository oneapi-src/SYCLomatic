add_custom_target(dpct-codepin-report-files ALL DEPENDS  ${CMAKE_CURRENT_SOURCE_DIR}/codepin-report.py
                    COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_SOURCE_DIR}/codepin-report.py
                        ${CMAKE_BINARY_DIR}/bin)

install(PROGRAMS codepin-report.py
    DESTINATION ./bin
    COMPONENT dpct-codepin-report)

add_llvm_install_targets(install-dpct-codepin-report
                        DEPENDS dpct-codepin-report-files
                        COMPONENT dpct-codepin-report)
