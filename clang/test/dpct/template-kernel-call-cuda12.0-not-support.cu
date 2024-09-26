// FIXME
// UNSUPPORTED: system-windows
// UNSUPPORTED: cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6
// UNSUPPORTED: v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6
// RUN: dpct --format-range=none --usm-level=none -out-root %T/template-kernel-call-cuda12.0-not-support %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only -std=c++11
// RUN: FileCheck --input-file %T/template-kernel-call-cuda12.0-not-support/template-kernel-call-cuda12.0-not-support.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl %T/template-kernel-call-cuda12.0-not-support/template-kernel-call-cuda12.0-not-support.dp.cpp -o %T/template-kernel-call-cuda12.0-not-support/template-kernel-call-cuda12.0-not-support.dp.o %}

texture<float4, 1, cudaReadModeElementType> posTexture;
texture<int4, 1, cudaReadModeElementType> posTexture_dp;
struct texReader_sp {
// CHECK:     __dpct_inline__ sycl::float4 operator()(int idx,
// CHECK-NEXT:              dpct::image_accessor_ext<sycl::float4, 1> posTexture) const
   __device__ __forceinline__ float4 operator()(int idx) const
   {
       return tex1Dfetch(posTexture, idx);
   }
};
struct texReader_dp {
// CHECK:   __dpct_inline__ sycl::double4 operator()(int idx,
// CHECK-NEXT:              dpct::image_accessor_ext<sycl::int4, 1> posTexture_dp) const
   __device__ __forceinline__ double4 operator()(int idx) const
   {
       int4 v = tex1Dfetch(posTexture_dp, idx*2);
       return make_double4(1, 1, 1, 1);
   }
};

template <typename texReader>
__global__ void compute_lj_force()
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    texReader positionTexReader;
    // CHECK: /*
    // CHECK: DPCT1084:{{[0-9]+}}: The function call "texReader_sp::operator()" has multiple migration results in different template instantiations that could not be unified. You may need to adjust the code.
    // CHECK: */
    float j = positionTexReader(idx).x;
}

void compute_lj_force_h() {
  compute_lj_force<texReader_sp><<<1,1>>>();
  compute_lj_force<texReader_dp><<<1,1>>>();
}

texture<int4, 1, cudaReadModeElementType> tex_1;
texture<int4, 1, cudaReadModeElementType> tex_2;
struct tex_reader_1 {
  __device__ int4 operator()(int idx) const { return tex1Dfetch(tex_1, idx); }
};
struct tex_reader_2 {
  __device__ int4 operator()(int idx) const { return tex1Dfetch(tex_1, idx); }
};
template <typename tex_reader> __global__ void kernel_2() {
  //CHECK:int idx = item_ct1.get_local_id(2);
  //CHECK-NEXT:tex_reader reader;
  //CHECK-NEXT:float res = reader(idx, tex_1).x();
  int idx = threadIdx.x;
  tex_reader reader;
  float res = reader(idx).x;
}
void foo_2() {
  kernel_2<tex_reader_1><<<1, 1>>>();
  kernel_2<tex_reader_2><<<1, 1>>>();
}
