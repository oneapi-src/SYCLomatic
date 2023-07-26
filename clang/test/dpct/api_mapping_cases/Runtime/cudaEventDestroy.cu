// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2, cuda-11.0, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2, v11.0, v11.1, v11.2, v11.3, v11.4, v11.5
// RUN: dpct --format-range=none -out-root %T/api_mapping_cases/Runtime %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only --std=c++14 2>&1 | FileCheck %s
// CHECK-NOT: {{.*}}error{{.*}}

void test(cudaEvent_t e) { cudaEventDestroy(e /*cudaEvent_t*/); }
