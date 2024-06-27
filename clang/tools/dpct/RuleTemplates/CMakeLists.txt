set(dpct_rule_template_files
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/RuleTemplates/api_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/RuleTemplates/class_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/RuleTemplates/disable_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/RuleTemplates/enum_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/RuleTemplates/header_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/RuleTemplates/macro_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/RuleTemplates/pattern_rewrite_rule.yaml
  ${CMAKE_SOURCE_DIR}/../clang/tools/dpct/RuleTemplates/type_rule.yaml
)

install(
  FILES ${dpct_rule_template_files}
  COMPONENT dpct-opt-rules
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ./extensions/rule_templates)