#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

void merge_by_key_test() {
  // clang-format off
  // Start
  thrust::host_vector<int> AH(4);
  thrust::host_vector<int> BH(4);
  thrust::host_vector<int> CH(4);
  thrust::host_vector<int> DH(4);
  thrust::host_vector<int> EH(8);
  thrust::host_vector<int> FH(8);
  /*1*/ thrust::merge_by_key(AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(),DH.begin(), EH.begin(), FH.begin());
  /*2*/ thrust::merge_by_key(AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(),DH.begin(), EH.begin(), FH.begin(), thrust::greater<int>());
  /*3*/ thrust::merge_by_key(thrust::host, AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin());
  /*4*/ thrust::merge_by_key(thrust::host, AH.begin(), AH.end(), BH.begin(), BH.end(), CH.begin(), DH.begin(), EH.begin(), FH.begin(), thrust::greater<int>());
  // End
  // clang-format on
}
