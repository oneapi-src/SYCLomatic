
list(APPEND dpct-codepin-auto-compare-dependends ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/codepin/auto-compare)

add_custom_target(dpct-codepin-auto-compare-files ALL DEPENDS ${dpct-codepin-auto-compare-dependends})

install(PROGRAMS auto-compare
        DESTINATION ./bin
        COMPONENT dpct-codepin-auto-compare)

add_llvm_install_targets(install-dpct-codepin-auto-compare
                        DEPENDS dpct-codepin-auto-compare-files
                        COMPONENT dpct-codepin-auto-compare)