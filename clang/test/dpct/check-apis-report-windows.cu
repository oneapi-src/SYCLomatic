// UNSUPPORTED: -linux-
// RUN: dpct -report-type=apis -report-file-prefix=check_apis_report -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: echo "// CHECK: API name, Frequency" >%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: CUresult cuMemGetInfo_v2(size_t * free,size_t * total),1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: cudaDeviceProp,1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: cudaError_t,3" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: cudaError_t cudaDeviceSynchronize(),4" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: cudaError_t cudaFree(void * devPtr),1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: cudaError_t cudaFreeHost(void * ptr),1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes * attr,void (*)(float *, const float *, const float *) entry),1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: cudaError_t cudaMalloc(void ** devPtr,size_t size),2" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: cudaError_t cudaMalloc3D(struct cudaPitchedPtr * pitchedDevPtr,struct cudaExtent extent),1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: cudaError_t cudaMallocHost(float ** ptr,size_t size,unsigned int flags),1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: cudaError_t cudaMemcpy(void * dst,const void * src,size_t count,enum cudaMemcpyKind kind),2" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: cudaError_t cudaMemset(void * devPtr,int value,size_t count),1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: cudaStream_t,1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: dim3,1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: float max(float a,float b),1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: int2,3" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: longlong4,1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: struct cudaChannelFormatDesc cudaCreateChannelDesc(),1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: struct cudaExtent make_cudaExtent(size_t w,size_t h,size_t d),2" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: struct cudaPitchedPtr make_cudaPitchedPtr(void * d,size_t p,size_t xsz,size_t ysz),1" >>%T/check_apis_report_csv_check.txt
// RUN: echo "// CHECK: uint4,1" >>%T/check_apis_report_csv_check.txt
// RUN: cat %T/check_apis_report.apis.csv >>%T/check_apis_report_csv_check.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_apis_report_csv_check.txt %T/check_apis_report_csv_check.txt

// RUN: dpct -report-file-prefix=report -report-type=apis  -report-format=formatted -report-only  -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: echo "// CHECK: API name				Frequency" >%T/check_apis_report_check.txt
// RUN: echo "// CHECK: CUresult cuMemGetInfo_v2(size_t * free,size_t * total)               1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: cudaDeviceProp                               1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: cudaError_t                                  3" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: cudaError_t cudaDeviceSynchronize()               4" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: cudaError_t cudaFree(void * devPtr)               1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: cudaError_t cudaFreeHost(void * ptr)               1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes * attr,void (*)(float *, const float *, const float *) entry)               1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: cudaError_t cudaMalloc(void ** devPtr,size_t size)               2" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: cudaError_t cudaMalloc3D(struct cudaPitchedPtr * pitchedDevPtr,struct cudaExtent extent)               1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: cudaError_t cudaMallocHost(float ** ptr,size_t size,unsigned int flags)               1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: cudaError_t cudaMemcpy(void * dst,const void * src,size_t count,enum cudaMemcpyKind kind)               2" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: cudaError_t cudaMemset(void * devPtr,int value,size_t count)               1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: cudaStream_t                                 1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: dim3                                         1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: float max(float a,float b)                   1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: int2                                         3" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: longlong4                                    1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: struct cudaChannelFormatDesc cudaCreateChannelDesc()               1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: struct cudaExtent make_cudaExtent(size_t w,size_t h,size_t d)               2" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: struct cudaPitchedPtr make_cudaPitchedPtr(void * d,size_t p,size_t xsz,size_t ysz)               1" >>%T/check_apis_report_check.txt
// RUN: echo "// CHECK: uint4                                        1" >>%T/check_apis_report_check.txt
// RUN: cat %T/report.apis.log >>%T/check_apis_report_check.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_apis_report_check.txt %T/check_apis_report_check.txt

// RUN: dpct -output-file=output-file.txt -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: echo "// CHECK: Processing: {{(.+)}}" >%T/check_output-file.txt
// RUN: echo "// CHECK: {{(.+)}} warning: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code." >>%T/check_output-file.txt
// RUN: echo "// CHECK:   cudaError_t err = cudaDeviceSynchronize();" >>%T/check_output-file.txt
// RUN: echo "// CHECK: {{(.+)}} warning: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code." >>%T/check_output-file.txt
// RUN: echo "// CHECK:   checkError(cudaDeviceSynchronize());" >>%T/check_output-file.txt
// RUN: echo "// CHECK: {{(.+)}} warning: DPCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may need to rewrite this code." >>%T/check_output-file.txt
// RUN: echo "// CHECK:   return cudaDeviceSynchronize();" >>%T/check_output-file.txt
// RUN: cat %T/output-file.txt >>%T/check_output-file.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_output-file.txt %T/check_output-file.txt

// NOMATCH-CHECK-NOT: '{{.}}'

// RUN: dpct -output-verbosity=silent  -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only  2>&1  \
// RUN: | FileCheck -check-prefix=NOMATCH-CHECK -allow-empty %s


// FAKE-FILE-CHECK-NOT:Processing: {{(.+)}}
// FAKE-FILE-STDERR: Processing: {{(.+)}}

// RUN: dpct -output-verbosity=normal  -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only  2>&1  \
// RUN: | FileCheck -check-prefix=FAKE-FILE-CHECK -allow-empty %s

// RUN: dpct -output-verbosity=detailed  -out-root %T %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only  2>&1  \
// RUN: | FileCheck -check-prefix=FAKE-FILE-STDERR -allow-empty %s

#include <cuda_runtime.h>

void checkError(cudaError_t err) {
}

void fooo() {
  size_t size = 10 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;

  size_t length = size * size * size;
  size_t bytes = length * sizeof(float);
  float *src;

  cudaFreeHost(d_A);

  cudaMallocHost(&src, bytes);

  struct cudaPitchedPtr srcGPU;

  struct cudaExtent extent = make_cudaExtent(size * sizeof(float), size, size);

  cudaMalloc3D(&srcGPU, extent);

  int2 a;
  uint4 b;
  dim3 d3;
  cudaDeviceProp cdp;
  cudaStream_t cuSt;
  const int2 c = {0,0};
  int2 d[100];
  longlong4 ll4;
}

int cool() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  cudaMalloc((void **)&d_A, size);
  cudaMemset(d_A, 0xf, size);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  free(h_A);
  cudaDeviceSynchronize();
  cudaError_t err = cudaDeviceSynchronize();
  checkError(cudaDeviceSynchronize());
  return cudaDeviceSynchronize();
}

static texture<float, 3, cudaReadModeElementType>
    NoiseTextures[4]; // texture Array

void CreateTexture();

__global__ void
AccesTexture(texture<float, 3, cudaReadModeElementType> *NoiseTextures) {
  int test = tex3D(NoiseTextures[0], threadIdx.x, threadIdx.y,
                   threadIdx.z); // by using this the error occurs
}

int main(int argc, char **argv) {
  AccesTexture<<<1, dim3(4, 4, 4)>>>(NoiseTextures);
}


template<typename T>
__global__ void addKernel(T *c, const T *a, const T *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

#define  SIZE_X 128 //numbers in elements
#define  SIZE_Y 128
#define  SIZE_Z 128
void bar(){
  typedef float  VolumeType;
  cudaExtent volumeSize = make_cudaExtent(SIZE_X, SIZE_Y, SIZE_Z);
  float d_volumeMem[100];
  cudaMalloc((void**)&d_volumeMem[0], SIZE_X*SIZE_Y*SIZE_Z*sizeof(float));

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
  make_cudaPitchedPtr((void*)d_volumeMem, SIZE_X*sizeof(VolumeType), SIZE_X, SIZE_Y);
  cudaFuncAttributes attrib;
  cudaError_t err;
  err = cudaFuncGetAttributes(&attrib, addKernel<float>);
}

namespace libsvm {
extern "C"
void SVMTrain(void){
    float* d_value_inter;
    size_t free_mem, total;
    cuMemGetInfo_v2(&free_mem, &total);
    int a = max(1, 3);
}
}
