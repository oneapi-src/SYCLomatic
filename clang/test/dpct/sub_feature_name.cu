// RUN: dpct --format-range=none -in-root=%S -out-root %T/sub_feature_name %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck %s --match-full-lines --input-file %T/sub_feature_name/MainSourceFiles.yaml

//CHECK:FeatureName:     'dev_mgr::check_id'
//CHECK:FeatureName:     'dev_mgr::current_device'
//CHECK:FeatureName:     'dev_mgr::current_device_id'
//CHECK:FeatureName:     'device_ext::default_queue'

__global__ void kernel() {
  extern __shared__ float3 shared_memory[];
}

int main() {
  size_t shared_size = 3 * sizeof(float);
  kernel<<<1, 1, shared_size>>>();
  return 0;
}
