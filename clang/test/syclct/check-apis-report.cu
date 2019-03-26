// RUN: syclct -report-type=apis -report-file-prefix=check-apis-report -out-root %T %s  -- -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: echo "// `perl -e 'print "CH","ECK"'`: API name, Migrated, Frequency" >%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaFree,true,1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaFreeHost,false,1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaMalloc,true,1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaMalloc3D,false,1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaMallocHost,false,1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaMemcpy,true,2" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: cudaMemset,true,1" >>%T/check-apis-report_check.txt
// RUN: echo "// `perl -e 'print "CH","ECK-NEXT"'`: make_cudaExtent,false,1" >>%T/check-apis-report_check.txt
// RUN: cat %T/check-apis-report.apis.csv >>%T/check-apis-report_check.txt
// RUN: FileCheck --match-full-lines --input-file %T/check-apis-report_check.txt %T/check-apis-report_check.txt

#include <cuda_runtime.h>

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
}

void cool() {
  size_t size = 1234567 * sizeof(float);
  float *h_A = (float *)malloc(size);
  float *d_A = NULL;
  cudaMalloc((void **)&d_A, size);
  cudaMemset(d_A, 0xf, size);
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  cudaFree(d_A);
  free(h_A);
}
