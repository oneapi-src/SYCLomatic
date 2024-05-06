// RUN: dpct --format-range=none --enable-codepin -out-root %T/debug_test/all_public_dump %s --cuda-include-path="%cuda-path/include" -- -std=c++17  -x cuda --cuda-host-only
// RUN: FileCheck %S/codepin_autogen_util.hpp.ref --match-full-lines --input-file %T/debug_test/all_public_dump_codepin_sycl/codepin_autogen_util.hpp
// RUN: FileCheck %S/codepin_autogen_util.hpp.cuda.ref --match-full-lines --input-file %T/debug_test/all_public_dump_codepin_cuda/codepin_autogen_util.hpp
// RUN: %if build_lit %{icpx -c -fsycl %T/debug_test/all_public_dump_codepin_sycl/test.dp.cpp -o %T/debug_test/all_public_dump_codepin_sycl/test.dp.o %}
#include <cuda.h>
#include <iostream>

struct Point2D{
  int x;
  int y;
};

template<typename T>
class Point3D {
public:
  T x;
  T y;
  T z;
};

struct Color {
  int r;
  int g;
  int b;
};

class Point3DExt : public Point3D<int> {
public:
  Color col;
};

struct PointCloud {
  float3 pc[3];
};

__global__ void kernel2d(Point2D* a, Point2D* b, Point2D* c) {
  int i = threadIdx.x;
  c[i].x = a[i].x + b[i].x;
  c[i].y = a[i].y + b[i].y;
}

template<typename T>
__global__ void kernel3d(Point3D<T>* a, Point3D<T>* b, Point3D<T>* c) {
  int i = threadIdx.x;
  c[i].x = a[i].x + b[i].x;
  c[i].y = a[i].y + b[i].y;
  c[i].z = a[i].z + b[i].z;
}

__global__ void kernel3dext(Point3DExt* a, Point3DExt* b, Point3DExt* c) {
  int i = threadIdx.x;
  c[i].x = a[i].x + b[i].x;
  c[i].y = a[i].y + b[i].y;
  c[i].z = a[i].z + b[i].z;
  c[i].col.r = a[i].col.r + b[i].col.r;
  c[i].col.g = a[i].col.g + b[i].col.g;
  c[i].col.b = a[i].col.b + b[i].col.b;
}

__global__ void kernelpc(PointCloud* a, PointCloud* b, PointCloud* c) {
  int i = threadIdx.x;
  for(int j = 0; j < 3; j++) {
    c[i].pc[j].x = a[i].pc[j].x + b[i].pc[j].x;
    c[i].pc[j].y = a[i].pc[j].y + b[i].pc[j].y;
    c[i].pc[j].z = a[i].pc[j].z + b[i].pc[j].z;
  }
}

#define NUM 10

int main() {
  Point2D h_2d[NUM];
  for(int i = 0; i < NUM; i++) {
    h_2d[i].x = i;
    h_2d[i].y = i;
  }  
  Point2D *d_a2d, *d_b2d, *d_c2d;	
  cudaMalloc(&d_a2d, sizeof(Point2D) * NUM);
  cudaMalloc(&d_b2d, sizeof(Point2D) * NUM);
  cudaMalloc(&d_c2d, sizeof(Point2D) * NUM);
  cudaMemcpy(d_a2d, h_2d, sizeof(Point2D) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2d, h_2d, sizeof(Point2D) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c2d, h_2d, sizeof(Point2D) * NUM, cudaMemcpyHostToDevice);
  kernel2d<<<1, NUM>>>(d_a2d, d_b2d, d_c2d);
  cudaDeviceSynchronize();

  Point3D<float> h_3d[NUM];
  for(int i = 0; i < NUM; i++) {
    h_3d[i].x = i;
    h_3d[i].y = i;
    h_3d[i].z = i;
  }
  Point3D<float> *d_a3d, *d_b3d, *d_c3d;	
  cudaMalloc(&d_a3d, sizeof(Point3D<float>) * NUM);
  cudaMalloc(&d_b3d, sizeof(Point3D<float>) * NUM);
  cudaMalloc(&d_c3d, sizeof(Point3D<float>) * NUM);
  cudaMemcpy(d_a3d, h_3d, sizeof(Point3D<float>) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b3d, h_3d, sizeof(Point3D<float>) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c3d, h_3d, sizeof(Point3D<float>) * NUM, cudaMemcpyHostToDevice);
  kernel3d<float><<<1, NUM>>>(d_a3d, d_b3d, d_c3d);
  cudaDeviceSynchronize();

  Point3DExt h_3dext[NUM];
  for(int i = 0; i < NUM; i++) {
    h_3dext[i].x = i;
    h_3dext[i].y = i;
    h_3dext[i].z = i;
    h_3dext[i].col.r = i;
    h_3dext[i].col.g = i;
    h_3dext[i].col.b = i;
  }
  Point3DExt *d_a3dext, *d_b3dext, *d_c3dext;	
  cudaMalloc(&d_a3dext, sizeof(Point3DExt) * NUM);
  cudaMalloc(&d_b3dext, sizeof(Point3DExt) * NUM);
  cudaMalloc(&d_c3dext, sizeof(Point3DExt) * NUM);
  cudaMemcpy(d_a3dext, h_3dext, sizeof(Point3DExt) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b3dext, h_3dext, sizeof(Point3DExt) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c3dext, h_3dext, sizeof(Point3DExt) * NUM, cudaMemcpyHostToDevice);
  kernel3dext<<<1, NUM>>>(d_a3dext, d_b3dext, d_c3dext);
  cudaDeviceSynchronize();

  PointCloud h_pc[NUM];
  for(int i = 0; i < NUM; i++) {
    for(int j = 0; j < 3; j++) {
      h_pc[i].pc[j].x = i;
      h_pc[i].pc[j].y = i;
      h_pc[i].pc[j].z = i;
    }
  }
  PointCloud *d_apc, *d_bpc, *d_cpc;
  cudaMalloc(&d_apc, sizeof(PointCloud) * NUM);
  cudaMalloc(&d_bpc, sizeof(PointCloud) * NUM);
  cudaMalloc(&d_cpc, sizeof(PointCloud) * NUM);
  cudaMemcpy(d_apc, h_pc, sizeof(PointCloud) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bpc, h_pc, sizeof(PointCloud) * NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cpc, h_pc, sizeof(PointCloud) * NUM, cudaMemcpyHostToDevice);
  kernelpc<<<1, NUM>>>(d_apc, d_bpc, d_cpc);
  cudaDeviceSynchronize();

  return 0;
}
