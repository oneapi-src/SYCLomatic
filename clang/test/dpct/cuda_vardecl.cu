// RUN: dpct -out-root %T/cuda_vardecl %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda_vardecl/cuda_vardecl.dp.cpp --match-full-lines %s

#include <vector>
#include <list>

// CHECK: sycl::event gv;
cudaEvent_t gv;
// CHECK: sycl::event *gp = &gv;
cudaEvent_t *gp = &gv;
// CHECK: sycl::event &gr = *gp;
cudaEvent_t &gr = *gp;

// CHECK: sycl::event *gp2, gv2;
cudaEvent_t *gp2, gv2;
// CHECK: sycl::event gv3, *gp3;
cudaEvent_t gv3, *gp3;
// CHECK: sycl::event gv4, &gr2 = gv2;
cudaEvent_t gv4, &gr2 = gv2;
// CHECK: sycl::event &gr3 = gv3, gv5;
cudaEvent_t &gr3 = gv3, gv5;
// CHECK: sycl::event &gr4 = gr3, *gp4;
cudaEvent_t &gr4 = gr3, *gp4;
// CHECK: sycl::event *gp5, &gr5 = gr4;
cudaEvent_t *gp5, &gr5 = gr4;
// CHECK: sycl::event gv6, *gp6, &gr6 = gr5;
cudaEvent_t gv6, *gp6, &gr6 = gr5;

// CHECK: sycl::event eventArray[23];
cudaEvent_t eventArray[23];
// CHECK: std::vector<sycl::event> eventVector;
std::vector<cudaEvent_t> eventVector;

// CHECK: void foo(sycl::event paramV, sycl::event *paramP, sycl::event &paramR) {
void foo(cudaEvent_t paramV, cudaEvent_t *paramP, cudaEvent_t &paramR) {
}

template <typename T1, typename T2>
class C {
};

struct S {
  // CHECK: sycl::event sv;
  cudaEvent_t sv;
  // CHECK: sycl::event *sp = &sv;
  cudaEvent_t *sp = &sv;
  // CHECK: sycl::event &sr = *sp;
  cudaEvent_t &sr = *sp;

  // CHECK: sycl::event *sp2, sv2;
  cudaEvent_t *sp2, sv2;
  // CHECK: sycl::event sv3, *sp3;
  cudaEvent_t sv3, *sp3;
  // CHECK: sycl::event sv4, &sr2 = sv2;
  cudaEvent_t sv4, &sr2 = sv2;
  // CHECK: sycl::event &sr3 = sv3, sv5;
  cudaEvent_t &sr3 = sv3, sv5;
  // CHECK: sycl::event &sr4 = sr3, *sp4;
  cudaEvent_t &sr4 = sr3, *sp4;
  // CHECK: sycl::event *sp5, &sr5 = sr4;
  cudaEvent_t *sp5, &sr5 = sr4;
  // CHECK: sycl::event sv6, *sp6, &sr6 = sr5;
  cudaEvent_t sv6, *sp6, &sr6 = sr5;

  // CHECK: sycl::event eventArray[23];
  cudaEvent_t eventArray[23];
  // CHECK: std::vector<sycl::event> eventVector;
  std::vector<cudaEvent_t> eventVector;

  // CHECK: sycl::queue *stream, *stream0;
  // CHECK-NEXT: sycl::queue *streams[23], *streams0[45];
  // CHECK-NEXT: sycl::event event, events[23];
  // CHECK-NEXT: C<sycl::queue *, sycl::event> se, se2;
  // CHECK-NEXT: std::list<sycl::queue *> streamlist;
  // CHECK-NEXT: std::list<sycl::event> eventlist;
  // CHECK-NEXT: std::list<int> errors;
  // CHECK-NEXT: std::list<dpct::device_info> props;
  cudaStream_t stream, stream0;
  cudaStream_t streams[23], streams0[45];
  cudaEvent_t event, events[23];
  C<cudaStream_t, cudaEvent_t> se, se2;
  std::list<cudaStream_t> streamlist;
  std::list<cudaEvent_t> eventlist;
  std::list<cudaError> errors;
  std::list<cudaDeviceProp> props;
};

int main(int argc, char* argv[]) {
  // CHECK: sycl::event v;
  cudaEvent_t v;
  // CHECK: sycl::event *p;
  cudaEvent_t *p;
  // CHECK: sycl::event **p1;
  cudaEvent_t **p1;
  // CHECK: sycl::event *p2 = &v;
  cudaEvent_t *p2 = &v;
  // CHECK: sycl::event **p3 = &p2;
  cudaEvent_t **p3 = &p2;
  // CHECK: sycl::event &r = v;
  cudaEvent_t &r = v;

  // CHECK: sycl::event vv, vv2;
  cudaEvent_t vv, vv2;
  // CHECK: sycl::event vv3, *pp;
  cudaEvent_t vv3, *pp;
  // CHECK: sycl::event *pp2, vv4;
  cudaEvent_t *pp2, vv4;
  // CHECK: sycl::event *pp3, *pp4;
  cudaEvent_t *pp3, *pp4;
  // CHECK: sycl::event vv5, &rr = vv;
  cudaEvent_t vv5, &rr = vv;
  // CHECK: sycl::event &rr2 = vv2, vv6;
  cudaEvent_t &rr2 = vv2, vv6;
  // CHECK: sycl::event *pp5, &rr3 = *pp;
  cudaEvent_t *pp5, &rr3 = *pp;
  // CHECK: sycl::event &rr4 = *pp5, *pp6 = &vv3;
  cudaEvent_t &rr4 = *pp5, *pp6 = &vv3;

  // CHECK: sycl::event vvv, vvv2, vvv3;
  cudaEvent_t vvv, vvv2, vvv3;
  // CHECK: sycl::event *ppp, *ppp2, *ppp3;
  cudaEvent_t *ppp, *ppp2, *ppp3;
  // CHECK: sycl::event &rrr = vvv, &rrr2 = vvv2, &rrr3 = vvv3;
  cudaEvent_t &rrr = vvv, &rrr2 = vvv2, &rrr3 = vvv3;
  // CHECK: sycl::event vvv4, *ppp4, &rrr4 = vvv;
  cudaEvent_t vvv4, *ppp4, &rrr4 = vvv;
  // CHECK: sycl::event *ppp5, vvv5, &rrr5 = vvv;
  cudaEvent_t *ppp5, vvv5, &rrr5 = vvv;
  // CHECK: sycl::event &rrr6 = vvv, vvv6, *ppp6;
  cudaEvent_t &rrr6 = vvv, vvv6, *ppp6;

  // CHECK: sycl::queue *stream, *stream0;
  // CHECK-NEXT: sycl::queue *streams[23], *streams0[45];
  // CHECK-NEXT: sycl::event event, events[23];
  // CHECK-NEXT: C<sycl::queue *, sycl::event> se, se2;
  // CHECK-NEXT: std::list<sycl::queue *> streamlist;
  // CHECK-NEXT: std::list<sycl::event> eventlist;
  // CHECK-NEXT: std::list<int> errors;
  // CHECK-NEXT: std::list<dpct::device_info> props;
  cudaStream_t stream, stream0;
  cudaStream_t streams[23], streams0[45];
  cudaEvent_t event, events[23];
  C<cudaStream_t, cudaEvent_t> se, se2;
  std::list<cudaStream_t> streamlist;
  std::list<cudaEvent_t> eventlist;
  std::list<cudaError> errors;
  std::list<cudaDeviceProp> props;

  // CHECK: std::vector<sycl::float2> const vf(5);
  std::vector<float2> const vf(5);
}

