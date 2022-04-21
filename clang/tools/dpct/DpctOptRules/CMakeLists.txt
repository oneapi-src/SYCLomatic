set(dpct_opt_rule_files
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/DpctOptRules/forceinline.yaml
)
install(
  FILES ${dpct_opt_rule_files}
  COMPONENT dpct-opt-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ./extensions/opt_rules)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-opt-rules
                           COMPONENT dpct-opt-rules)
endif()
