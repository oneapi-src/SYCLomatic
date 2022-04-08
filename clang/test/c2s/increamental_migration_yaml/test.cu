// RUN: c2s --format-range=none --out-root %T/increamental_migration_yaml/out %s --use-custom-helper=api --custom-helper-name=test --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -DAAA
// RUN: c2s --format-range=none --out-root %T/increamental_migration_yaml/out %s --use-custom-helper=api --custom-helper-name=test --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/increamental_migration_yaml/out/MainSourceFiles.yaml --match-full-lines %s
// RUN: rm -rf %T/increamental_migration_yaml/out

//     CHECK:FeatureMap:
//CHECK-NEXT:  test.hpp:
//CHECK-NEXT:    c2s_align_and_inline:
//CHECK-NEXT:      IsCalled:        true
//CHECK-NEXT:      CallerSrcFiles:
//CHECK-NEXT:        - '{{(.+)}}'
//CHECK-NEXT:      FeatureName:     '__c2s_align__(n) and __c2s_inline__'
//CHECK-NEXT:      SubFeatureMap:   {}
//CHECK-NEXT:    c2s_compatibility_temp:
//CHECK-NEXT:      IsCalled:        true
//CHECK-NEXT:      CallerSrcFiles:
//CHECK-NEXT:        - '{{(.+)}}'
//CHECK-NEXT:      FeatureName:     C2S_COMPATIBILITY_TEMP
//CHECK-NEXT:      SubFeatureMap:   {}
//CHECK-NEXT:    non_local_include_dependency:
//CHECK-NEXT:      IsCalled:        true
//CHECK-NEXT:      CallerSrcFiles:
//CHECK-NEXT:        - ''
//CHECK-NEXT:      FeatureName:     ''
//CHECK-NEXT:      SubFeatureMap:   {}

#ifdef AAA
__forceinline__ void foo(){}
#else
#define BBB __CUDA_ARCH__
#endif

int main() {
  return 0;
}
