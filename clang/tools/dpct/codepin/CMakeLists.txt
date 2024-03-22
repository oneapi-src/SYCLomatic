ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_BINARY_DIR}/bin/auto-compare
                COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_CURRENT_SOURCE_DIR}/auto-compare
                        ${CMAKE_BINARY_DIR}/bin)
list(APPEND dpct-codepin-auto-compare-dependends ${CMAKE_BINARY_DIR}/bin/auto-compare)

add_custom_target(dpct-codepin-auto-compare-files ALL DEPENDS ${dpct-codepin-auto-compare-dependends})

install(PROGRAMS auto-compare
    DESTINATION ./bin
    COMPONENT dpct-codepin-auto-compare)

add_llvm_install_targets(install-dpct-codepin-auto-compare
                        DEPENDS dpct-codepin-auto-compare-files
                        COMPONENT dpct-codepin-auto-compare)
