// RUN: mkdir %T/inc_migration_with_different_analysis_path
// RUN: cd %T/inc_migration_with_different_analysis_path
// RUN: mkdir inner_folder
// RUN: cd inner_folder
// RUN: cp %s .
// RUN: dpct --out-root out inc_migration_with_different_analysis_path.cu --in-root .  --cuda-include-path="%cuda-path/include" --process-all
// RUN: dpct --out-root out inc_migration_with_different_analysis_path.cu --in-root ..  --cuda-include-path="%cuda-path/include" --process-all > log.txt 2>&1 || true
// RUN: grep -w "process-all" log.txt
// RUN: grep -w "analysis-scope-path" log.txt
// RUN: cd ../..
// RUN: rm -rf inc_migration_with_different_analysis_path

float2 f2;

