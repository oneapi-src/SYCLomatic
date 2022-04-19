// UNSUPPORTED: -linux-
// RUN: dpct -report-type=apis -report-file-prefix=check_apis_report -out-root %T/check_apis_report-windows %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: cat %S/check-apis-report_csv_ref_windows.txt > %T/check_apis_report-windows/check_apis_report_csv_check_windows.txt
// RUN: cat %T/check_apis_report-windows/check_apis_report.apis.csv >>%T/check_apis_report-windows/check_apis_report_csv_check_windows.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_apis_report-windows/check_apis_report_csv_check_windows.txt %T/check_apis_report-windows/check_apis_report_csv_check_windows.txt

// RUN: dpct  -output-file=output_file_all_windows.txt -report-type=apis -report-file-prefix=stdout -out-root %T/check_apis_report-windows %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: cat %S/check-apis-report_outputfile_ref_all_windows.txt > %T/check_apis_report-windows/check_output_file_all_windows.txt
// RUN: cat %T/check_apis_report-windows/output_file_all_windows.txt >>%T/check_apis_report-windows/check_output_file_all_windows.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_apis_report-windows/check_output_file_all_windows.txt %T/check_apis_report-windows/check_output_file_all_windows.txt

// RUN: dpct -report-file-prefix=report -report-type=apis  -report-format=formatted -report-only  -out-root %T/check_apis_report-windows %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: cat %S/check-apis-report_log_ref_windows.txt > %T/check_apis_report-windows/check_apis_report_check_windows.txt
// RUN: cat %T/check_apis_report-windows/report.apis.log >>%T/check_apis_report-windows/check_apis_report_check_windows.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_apis_report-windows/check_apis_report_check_windows.txt %T/check_apis_report-windows/check_apis_report_check_windows.txt

// RUN: dpct -output-file=output-file.txt -out-root %T/check_apis_report-windows %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
// RUN: cat %S/check-apis-report_outputfile_ref_window.txt > %T/check_apis_report-windows/check_outputfile_windows.txt
// RUN: cat %T/check_apis_report-windows/output-file.txt >>%T/check_apis_report-windows/check_outputfile_windows.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_apis_report-windows/check_outputfile_windows.txt %T/check_apis_report-windows/check_outputfile_windows.txt

// NOMATCH-CHECK-NOT: '{{.}}'

// RUN: dpct -output-verbosity=silent  -out-root %T/check_apis_report-windows %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only  2>&1  \
// RUN: | FileCheck -check-prefix=NOMATCH-CHECK -allow-empty %s


// FAKE-FILE-CHECK-NOT:Processing: {{(.+)}}
// FAKE-FILE-STDERR: Processing: {{(.+)}}

// RUN: dpct -output-verbosity=normal  -out-root %T/check_apis_report-windows %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only  2>&1  \
// RUN: | FileCheck -check-prefix=FAKE-FILE-CHECK -allow-empty %s

// RUN: dpct -output-verbosity=detailed  -out-root %T/check_apis_report-windows %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only  2>&1  \
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
    CUdeviceptr base, dptr;
    size_t size_range;
    cuMemGetAddressRange_v2(&base, &size_range, dptr);
    int a = max(1, 3);
}
}
