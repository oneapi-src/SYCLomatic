// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test25_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test25_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test25_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test25_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test25_out

// CHECK: 37
// TEST_FEATURE: DplExtrasAlgorithm_set_intersection

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/set_operations.h>
#include <thrust/execution_policy.h>

void test_1() {
    const int N=7,M=5,P=3;
    int Akey[N]={0, 1, 3, 4, 5, 6, 9};
    int Avalue[N]={0, 0, 0, 0, 0, 0, 0};
    int Bkey[M]={1, 3, 5, 7, 9};
    int Bvalue[N]={1, 1, 1, 1, 1 };
    int Ckey[P];
    int Cvalue[P];

    thrust::pair<int*,int*> result_end = thrust::set_difference_by_key(thrust::host,Akey,Akey+N,Bkey,Bkey+M,Avalue,Bvalue,Ckey,Cvalue);
}
