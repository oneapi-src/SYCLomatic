set(dpct_opt_rule_files
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/opt_rules/forceinline.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/opt_rules/intel_specific_math.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/opt_rules/macro_checks.yaml
)

set(dpct_cmake_rule_files
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/cmake_rules/cmake_script_migration_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/cmake_rules/cmake_script_migration_rule_optional.yaml
)

set(dpct_python_rule_files
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/python_rules/python_build_script_migration_rule_ipex.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/python_rules/python_build_script_migration_rule_pytorch.yaml
)

set(dpct_pytorch_api_rule_files
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/pytorch_api_rules/pytorch_api.yaml
)

set(dpct_rule_template_files
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/rule_templates/api_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/rule_templates/class_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/rule_templates/disable_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/rule_templates/enum_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/rule_templates/header_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/rule_templates/macro_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/rule_templates/pattern_rewrite_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/rule_templates/type_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/rule_templates/helper_function_rule.yaml
)

install(
  FILES ${dpct_opt_rule_files}
  COMPONENT dpct-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ./extensions/opt_rules)

install(
  FILES ${dpct_cmake_rule_files}
  COMPONENT dpct-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ./extensions/cmake_rules)

install(
  FILES ${dpct_python_rule_files}
  COMPONENT dpct-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ./extensions/python_rules)

install(
  FILES ${dpct_pytorch_api_rule_files}
  COMPONENT dpct-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ./extensions/pytorch_api_rules)

install(
  FILES ${dpct_rule_template_files}
  COMPONENT dpct-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ./extensions/rule_templates)

install(
  FILES ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/extensions/README.md
  COMPONENT dpct-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ./extensions)

# Below two install commands are copying those files into some location under
# the build folder, which need be accessed during the LIT.
install(
  FILES ${dpct_cmake_rule_files}
  COMPONENT dpct-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ${CMAKE_BINARY_DIR}/extensions/cmake_rules)
install(
  FILES ${dpct_python_rule_files}
  COMPONENT dpct-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ${CMAKE_BINARY_DIR}/extensions/python_rules)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_llvm_install_targets(install-dpct-rules
                           COMPONENT dpct-rules)
endif()
