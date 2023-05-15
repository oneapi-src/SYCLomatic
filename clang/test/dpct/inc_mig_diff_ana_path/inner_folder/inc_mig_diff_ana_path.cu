// RUN: cd %S
// RUN: dpct --out-root %T/inc_mig_diff_ana_path %s --in-root .  --cuda-include-path="%cuda-path/include" --process-all
// RUN: dpct --out-root %T/inc_mig_diff_ana_path %s --in-root ..  --cuda-include-path="%cuda-path/include" --process-all > %T/inc_mig_diff_ana_path/log.txt 2>&1 || true
// RUN: grep -w "process-all" %T/inc_mig_diff_ana_path/log.txt
// RUN: grep -w "analysis-scope-path" %T/inc_mig_diff_ana_path/log.txt

float2 f2;
