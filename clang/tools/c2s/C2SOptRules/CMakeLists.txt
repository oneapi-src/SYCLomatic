set(c2s_opt_rule_files
  ${CMAKE_SOURCE_DIR}/../clang/tools/c2s/C2SOptRules/forceinline.yaml
)
if(INTEL_DEPLOY_UNIFIED_LAYOUT)
  install(
    FILES ${c2s_opt_rule_files}
    COMPONENT c2s-opt-rules
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    DESTINATION ./share/c2s/extensions/opt_rules)
else()
  install(
    FILES ${c2s_opt_rule_files}
    COMPONENT c2s-opt-rules
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    DESTINATION ./extensions/opt_rules)
endif()

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-c2s-opt-rules
                           COMPONENT c2s-opt-rules)
endif()
