set(dpct_opt_rule_files
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/DpctOptRules/forceinline.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/DpctOptRules/intel_specific_math.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/DpctOptRules/macro_checks.yaml
)

set(dpct_cmake_rule_files
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/DpctOptRules/cmake_script_migration_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/DpctOptRules/cmake_script_migration_rule_optional.yaml
)

install(
  FILES ${dpct_opt_rule_files}
  COMPONENT dpct-opt-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ./extensions/opt_rules)

install(
  FILES ${dpct_cmake_rule_files}
  COMPONENT dpct-cmake-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ./extensions/cmake_rules)

install(
  FILES ${dpct_cmake_rule_files}
  COMPONENT dpct-cmake-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ${CMAKE_BINARY_DIR}/extensions/cmake_rules)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-opt-rules
                           COMPONENT dpct-opt-rules)
  add_llvm_install_targets(install-dpct-cmake-rules
                           COMPONENT dpct-cmake-rules)
endif()
