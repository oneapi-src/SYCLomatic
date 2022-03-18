set(dpct_opt_rule_files
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/DpctOptRules/forceinline.yaml
)
if(INTEL_DEPLOY_UNIFIED_LAYOUT)
  install(
    FILES ${dpct_opt_rule_files}
    COMPONENT dpct-opt-rules
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    DESTINATION ./share/dpct/extensions/opt_rules)
else()
  install(
    FILES ${dpct_opt_rule_files}
    COMPONENT dpct-opt-rules
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    DESTINATION ./extensions/opt_rules)
endif()

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-opt-rules
                           COMPONENT dpct-opt-rules)
endif()
