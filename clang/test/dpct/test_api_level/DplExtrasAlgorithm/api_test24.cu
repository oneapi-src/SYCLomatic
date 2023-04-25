// UNSUPPORTED: cuda-8.0
// UNSUPPORTED: v8.0
// RUN: dpct --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/DplExtrasAlgorithm/api_test24_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++17
// RUN: grep "IsCalled" %T/DplExtrasAlgorithm/api_test24_out/MainSourceFiles.yaml | wc -l > %T/DplExtrasAlgorithm/api_test24_out/count.txt
// RUN: FileCheck --input-file %T/DplExtrasAlgorithm/api_test24_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/DplExtrasAlgorithm/api_test24_out

// CHECK: 2
// TEST_FEATURE: DplExtrasAlgorithm_partition

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>


struct is_even
{
  __host__ __device__
  bool operator()(const int &x) const
  {
    return (x % 2) == 0;
  }
};


void test_1() {
    
    int datas[]={1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int ans[]={2, 4, 6, 8, 10, 1, 3, 5, 7, 9};
    const int N=sizeof(datas)/sizeof(int);
    thrust::host_vector<int> v(datas,datas+N);
   
    thrust::partition(thrust::host, v.begin(), v.end(),is_even());
}
