// RUN: c2s --out-root %T/group_local_memory_option_check %s --cuda-include-path="%cuda-path/include" --use-experimental-features=local-memory-kernel-scope-allocation
// RUN: c2s --out-root %T/group_local_memory_option_check %s --cuda-include-path="%cuda-path/include" 2>%T/group_local_memory_option_check/output.txt || true
// RUN: grep "use-experimental-features=local-memory-kernel-scope-allocation" %T/group_local_memory_option_check/output.txt | wc -l > %T/group_local_memory_option_check/wc_output.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/group_local_memory_option_check/wc_output.txt

// CHECK: 1

int main() {
  float2 f2;
  return 0;
}
