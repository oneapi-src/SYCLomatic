// RUN: dpct --format-range=none --out-root %T/increamental_migration_yaml/out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -DAAA
// RUN: dpct --format-range=none --out-root %T/increamental_migration_yaml/out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/increamental_migration_yaml/out/MainSourceFiles.yaml --match-full-lines %s
// RUN: rm -rf %T/increamental_migration_yaml/out

//     CHECK:FeatureMap: {}

#ifdef AAA
__forceinline__ void foo(){}
#else
#define BBB __CUDA_ARCH__
#endif

int main() {
  return 0;
}
