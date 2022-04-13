set(c2s_opt_rule_files
  ${CMAKE_SOURCE_DIR}/../clang/tools/c2s/C2SOptRules/forceinline.yaml
)
install_c2s_rule_files()
if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-c2s-opt-rules
                           COMPONENT c2s-opt-rules)
endif()
