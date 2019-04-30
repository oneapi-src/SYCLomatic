// RUN: syclct -report-type=apis -report-file-prefix=check-apis-report -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: echo "// `perl -e 'print "CH","ECK"'`: API name, Frequency" >%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaDeviceSynchronize(),4" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaFree(void * devPtr),1" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaMemcpy(void * dst,const void * src,size_t count,enum cudaMemcpyKind kind),2" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaMemset(void * devPtr,int value,size_t count),1" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: struct cudaPitchedPtr make_cudaPitchedPtr(void * d,size_t p,size_t xsz,size_t ysz),1" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaMalloc(void ** devPtr,size_t size),2" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: longlong4,1" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaMalloc3D(struct cudaPitchedPtr * pitchedDevPtr,struct cudaExtent extent),1" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaMallocHost(float ** ptr,size_t size,unsigned int flags),1" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: struct cudaExtent make_cudaExtent(size_t w,size_t h,size_t d),2" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes * attr,void (*)(float *, const float *, const float *) entry),1" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaStream_t,1" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: uint4,1" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaFreeHost(void * ptr),1" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: struct cudaChannelFormatDesc cudaCreateChannelDesc(),1" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: dim3,1" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t,3" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: int2,3" >>%T/check-apis-report_csv_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaDeviceProp,1" >>%T/check-apis-report_csv_check.txt
// RUN: cat %T/check-apis-report.apis.csv >>%T/check-apis-report_csv_check.txt
// RUN: FileCheck --match-full-lines --input-file %T/check-apis-report_csv_check.txt %T/check-apis-report_csv_check.txt

// RUN: syclct -report-file-prefix=report -report-type=apis  -report-format=formatted -report-only  -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: echo "// `perl -e 'print "CH","ECK"'`: API name				Frequency" >%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaDeviceSynchronize()               4" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaFree(void * devPtr)               1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaMemcpy(void * dst,const void * src,size_t count,enum cudaMemcpyKind kind)               2" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaMemset(void * devPtr,int value,size_t count)               1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: struct cudaPitchedPtr make_cudaPitchedPtr(void * d,size_t p,size_t xsz,size_t ysz)               1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaMalloc(void ** devPtr,size_t size)               2" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: longlong4                                    1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaMalloc3D(struct cudaPitchedPtr * pitchedDevPtr,struct cudaExtent extent)               1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaMallocHost(float ** ptr,size_t size,unsigned int flags)               1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: struct cudaExtent make_cudaExtent(size_t w,size_t h,size_t d)               2" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes * attr,void (*)(float *, const float *, const float *) entry)               1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaStream_t                                 1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: uint4                                        1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t cudaFreeHost(void * ptr)               1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: struct cudaChannelFormatDesc cudaCreateChannelDesc()               1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: dim3                                         1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaError_t                                  3" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: int2                                         3" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: cudaDeviceProp                               1" >>%T/check-apis-report_check.txt
// RUN: cat %T/report.apis.log >>%T/check-apis-report_check.txt
// RUN: FileCheck --match-full-lines --input-file %T/check-apis-report_check.txt %T/check-apis-report_check.txt

// RUN: syclct -output-file=output-file.txt -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: echo "// `perl -e 'print "CH","ECK"'`: Starting to parse: {{(.+)/([^/]+)}}" >%T/check_output-file.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: {{(.+)/([^/]+)}}:{{[0-9]+}}:{{[0-9]+}} warning: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code" >>%T/check_output-file.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`:   cudaError_t err = cudaDeviceSynchronize();" >>%T/check_output-file.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: {{(.+)/([^/]+)}}:{{[0-9]+}}:{{[0-9]+}} warning: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code" >>%T/check_output-file.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`:   checkError(cudaDeviceSynchronize());" >>%T/check_output-file.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: {{(.+)/([^/]+)}}:{{[0-9]+}}:{{[0-9]+}} warning: SYCLCT1003:{{[0-9]+}}: Migrated api does not return error code. (*, 0) is inserted. You may want to rewrite this code" >>%T/check_output-file.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`:   return cudaDeviceSynchronize();" >>%T/check_output-file.txt
// RUN: echo "// `perl -e 'print "CH","ECK"'`: Ending to parse: {{(.+)/([^/]+)}}" >>%T/check_output-file.txt
// RUN: cat %T/output-file.txt >>%T/check_output-file.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_output-file.txt %T/check_output-file.txt

// NOMATCH-CHECK-NOT: '{{.}}'

// RUN: syclct -output-verbosity=silent  -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path  2>&1  \
// RUN: | FileCheck -check-prefix=NOMATCH-CHECK -allow-empty %s


// FAKE-FILE-CHECK-NOT:Starting to parse: {{(.+)/([^/]+)}}
// FAKE-FILE-STDERR: Ending to parse: {{(.+)/([^/]+)}}

// RUN: syclct -output-verbosity=normal  -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path  2>&1  \
// RUN: | FileCheck -check-prefix=FAKE-FILE-CHECK -allow-empty %s

// RUN: syclct -output-verbosity=detailed  -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path  2>&1  \
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
