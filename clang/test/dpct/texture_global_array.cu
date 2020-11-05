// RUN: dpct --format-range=none -out-root %T/texture_global_array %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14 -fno-delayed-template-parsing
// RUN: FileCheck --input-file %T/texture_global_array/texture_global_array.dp.cpp --match-full-lines %s

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_INSTANCES 2

// CHECK: dpct::image_data resDescInput[MAX_INSTANCES];
// CHECK-NEXT: dpct::sampling_info texDescInput[MAX_INSTANCES];
// CHECK-NEXT: dpct::image_matrix *d_Input[MAX_INSTANCES] = {NULL};
// CHECK-NEXT: dpct::image_wrapper_base_p tex_Input[MAX_INSTANCES] = {NULL};
cudaResourceDesc resDescInput[MAX_INSTANCES];
cudaTextureDesc texDescInput[MAX_INSTANCES];
cudaArray *d_Input[MAX_INSTANCES] = {NULL};
cudaTextureObject_t tex_Input[MAX_INSTANCES] = {NULL};

void createTestTexture(int instance, unsigned char *d_In, int rSize, int pSize,
                       int pPitch) {
  // CHECK: memset(&resDescInput[instance], 0, sizeof(resDescInput[instance]));
  // CHECK-NEXT: resDescInput[instance].set_data_type(dpct::image_data_type::pitch);
  // CHECK-NEXT: resDescInput[instance].set_data_ptr(d_In);
  // CHECK-NEXT: dpct::image_channel channelDesc =
  // CHECK-NEXT:     /*
  // CHECK-NEXT:     DPCT1059:{{[0-9]+}}: SYCL only supports 4-channel image format. Adjust the code.
  // CHECK-NEXT:     */
  // CHECK-NEXT:     dpct::image_channel(8, 0, 0, 0, dpct::image_channel_data_type::unsigned_int);
  // CHECK-NEXT: resDescInput[instance].set_channel(channelDesc);
  // CHECK-NEXT: resDescInput[instance].set_y(pSize);
  // CHECK-NEXT: resDescInput[instance].set_x(rSize);
  // CHECK-NEXT: resDescInput[instance].set_pitch(pPitch);
  memset(&resDescInput[instance], 0, sizeof(resDescInput[instance]));
  resDescInput[instance].resType = cudaResourceTypePitch2D;
  resDescInput[instance].res.pitch2D.devPtr = d_In;
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
  resDescInput[instance].res.pitch2D.desc = channelDesc;
  resDescInput[instance].res.pitch2D.height = pSize;
  resDescInput[instance].res.pitch2D.width = rSize;
  resDescInput[instance].res.pitch2D.pitchInBytes = pPitch;

  // CHECK: memset(&texDescInput[instance], 0, sizeof(texDescInput[instance]));
  // CHECK-NEXT: texDescInput[instance].set(sycl::filtering_mode::linear);
  // CHECK-NEXT: texDescInput[instance].set(sycl::coordinate_normalization_mode::unnormalized);
  // CHECK-NEXT: texDescInput[instance].set(sycl::addressing_mode::clamp);
  // CHECK-NEXT: texDescInput[instance].set(sycl::addressing_mode::clamp);
  // CHECK-NEXT: if (tex_Input[instance] != NULL) {
  // CHECK-NEXT:   delete tex_Input[instance];
  // CHECK-NEXT:   tex_Input[instance] = NULL;
  // CHECK-NEXT: }
  // CHECK-NEXT: tex_Input[instance] = dpct::create_image_wrapper(resDescInput[instance], texDescInput[instance]);
  memset(&texDescInput[instance], 0, sizeof(texDescInput[instance]));
  texDescInput[instance].filterMode = cudaFilterModeLinear;
  texDescInput[instance].normalizedCoords = false;
  texDescInput[instance].addressMode[0] = cudaAddressModeBorder;
  texDescInput[instance].addressMode[1] = cudaAddressModeBorder;
  if (tex_Input[instance] != NULL) {
    cudaDestroyTextureObject(tex_Input[instance]);
    tex_Input[instance] = NULL;
  }
  cudaCreateTextureObject(&tex_Input[instance], &resDescInput[instance],
                          &texDescInput[instance], NULL);
}

void createTestTextureAlternative(int instance, unsigned char *d_In, int rSize,
                                  int pSize, int pPitch) {
  // CHECK: memset(&resDescInput[instance], 0, sizeof(resDescInput[instance]));
  // CHECK-NEXT: resDescInput[instance].set_data_type(dpct::image_data_type::matrix);
  // CHECK-NEXT: resDescInput[instance].set_data_ptr(d_Input[instance]);
  // CHECK-NEXT: memset(&texDescInput[instance], 0, sizeof(texDescInput[instance]));
  // CHECK-NEXT: texDescInput[instance].set(sycl::filtering_mode::linear);
  // CHECK-NEXT: texDescInput[instance].set(sycl::coordinate_normalization_mode::unnormalized);
  // CHECK-NEXT: texDescInput[instance].set(sycl::addressing_mode::clamp);
  // CHECK-NEXT: texDescInput[instance].set(sycl::addressing_mode::clamp);
  memset(&resDescInput[instance], 0, sizeof(resDescInput[instance]));
  resDescInput[instance].resType = cudaResourceTypeArray;
  resDescInput[instance].res.array.array = d_Input[instance];
  memset(&texDescInput[instance], 0, sizeof(texDescInput[instance]));
  texDescInput[instance].filterMode = cudaFilterModeLinear;
  texDescInput[instance].normalizedCoords = false;
  texDescInput[instance].addressMode[0] = cudaAddressModeBorder;
  texDescInput[instance].addressMode[1] = cudaAddressModeBorder;

  // CHECK: if (tex_Input[instance] != NULL) {
  // CHECK-NEXT:   delete tex_Input[instance];
  // CHECK-NEXT:   tex_Input[instance] = NULL;
  // CHECK-NEXT: }
  // CHECK-NEXT: tex_Input[instance] = dpct::create_image_wrapper(resDescInput[instance], texDescInput[instance]);
  if (tex_Input[instance] != NULL) {
    cudaDestroyTextureObject(tex_Input[instance]);
    tex_Input[instance] = NULL;
  }
  cudaCreateTextureObject(&tex_Input[instance], &resDescInput[instance],
                          &texDescInput[instance], NULL);
}

// CHECK: void test_Kernel(dpct::image_accessor_ext<float, 2> tex_inArg, float *d_out,
// CHECK-NEXT: int yPitchOutInFloat, sycl::nd_item<3> item_ct1) {
__global__ void test_Kernel(cudaTextureObject_t tex_inArg, float *d_out,
                            int yPitchOutInFloat) {
  // x and y are coordinates of the output 2D array
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y;
  int i = y * yPitchOutInFloat + x;

  float rTextureCoord = x;
  float pTextureCoord = y;
  // CHECK: float V = tex_inArg.read(rTextureCoord + 0.5f, pTextureCoord + 0.5f);
  float V = tex2D<float>(tex_inArg, rTextureCoord + 0.5f, pTextureCoord + 0.5f);

  d_out[i] = V;
}

void test(float *d_Out, int rSize, int pSize, int pPitch) {
  int numThreadsPerBlock = 128;
  int blocks = (rSize + numThreadsPerBlock - 1) / numThreadsPerBlock;
  dim3 blockSz(numThreadsPerBlock);
  dim3 gridSz(blocks, pSize);
  //CHECK: dpct::get_default_queue().submit(
  //CHECK-NEXT:   [&](sycl::handler &cgh) {
  //CHECK-NEXT:     auto tex_Input_0_acc = static_cast<dpct::image_wrapper<float, 2> *>(tex_Input[0])->get_access(cgh);
  //CHECK-EMPTY:
  //CHECK-NEXT:     auto tex_Input_0_smpl = tex_Input[0]->get_sampler();
  //CHECK-EMPTY:
  //CHECK-NEXT:     cgh.parallel_for(
  //CHECK-NEXT:       sycl::nd_range<3>(gridSz * blockSz, blockSz),
  //CHECK-NEXT:       [=](sycl::nd_item<3> item_ct1) {
  //CHECK-NEXT:         test_Kernel(dpct::image_accessor_ext<float, 2>(tex_Input_0_smpl, tex_Input_0_acc), d_Out, pPitch / sizeof(float), item_ct1);
  //CHECK-NEXT:       });
  //CHECK-NEXT:   });
  test_Kernel<<<gridSz, blockSz>>>(tex_Input[0], d_Out, pPitch / sizeof(float));
}
#undef MAX_INSTANCES

