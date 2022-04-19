// RUN: c2s --format-range=none  --usm-level=none  --use-custom-helper=api -out-root %T/Image/api_test29_out %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: grep "IsCalled" %T/Image/api_test29_out/MainSourceFiles.yaml | wc -l > %T/Image/api_test29_out/count.txt
// RUN: FileCheck --input-file %T/Image/api_test29_out/count.txt --match-full-lines %s
// RUN: rm -rf %T/Image/api_test29_out

// CHECK: 77
// TEST_FEATURE: Image_image_wrapper_get_access
// TEST_FEATURE: Image_image_wrapper_base_get_sampler

static texture<float4, cudaTextureType2DLayered> tex42;

__global__ void kernel(float4 *out) {
  out[1] = tex2DLayered(tex42, 1.0f, 1.0f, 12);
}

int main() {
  float4 *d;
  kernel<<<1, 1>>>(d);
  return 0;
}
