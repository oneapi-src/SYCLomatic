// RUN: dpct -out-root %T/cuda_vardecl %s --cuda-include-path="%cuda-path/include" -- -std=c++14 -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/cuda_vardecl/cuda_vardecl.dp.cpp --match-full-lines %s

#include <vector>
#include <list>

// CHECK: dpct::event_ptr gv;
cudaEvent_t gv;
// CHECK: dpct::event_ptr *gp = &gv;
cudaEvent_t *gp = &gv;
// CHECK: dpct::event_ptr &gr = *gp;
cudaEvent_t &gr = *gp;

// CHECK: dpct::event_ptr *gp2, gv2;
cudaEvent_t *gp2, gv2;
// CHECK: dpct::event_ptr gv3, *gp3;
cudaEvent_t gv3, *gp3;
// CHECK: dpct::event_ptr gv4, &gr2 = gv2;
cudaEvent_t gv4, &gr2 = gv2;
// CHECK: dpct::event_ptr &gr3 = gv3, gv5;
cudaEvent_t &gr3 = gv3, gv5;
// CHECK: dpct::event_ptr &gr4 = gr3, *gp4;
cudaEvent_t &gr4 = gr3, *gp4;
// CHECK: dpct::event_ptr *gp5, &gr5 = gr4;
cudaEvent_t *gp5, &gr5 = gr4;
// CHECK: dpct::event_ptr gv6, *gp6, &gr6 = gr5;
cudaEvent_t gv6, *gp6, &gr6 = gr5;

// CHECK: dpct::event_ptr eventArray[23];
cudaEvent_t eventArray[23];
// CHECK: std::vector<dpct::event_ptr> eventVector;
std::vector<cudaEvent_t> eventVector;

// CHECK: void foo(dpct::event_ptr paramV, dpct::event_ptr *paramP, dpct::event_ptr &paramR) {
void foo(cudaEvent_t paramV, cudaEvent_t *paramP, cudaEvent_t &paramR) {
}

template <typename T1, typename T2>
class C {
};

struct S {
  // CHECK: dpct::event_ptr sv;
  cudaEvent_t sv;
  // CHECK: dpct::event_ptr *sp = &sv;
  cudaEvent_t *sp = &sv;
  // CHECK: dpct::event_ptr &sr = *sp;
  cudaEvent_t &sr = *sp;

  // CHECK: dpct::event_ptr *sp2, sv2;
  cudaEvent_t *sp2, sv2;
  // CHECK: dpct::event_ptr sv3, *sp3;
  cudaEvent_t sv3, *sp3;
  // CHECK: dpct::event_ptr sv4, &sr2 = sv2;
  cudaEvent_t sv4, &sr2 = sv2;
  // CHECK: dpct::event_ptr &sr3 = sv3, sv5;
  cudaEvent_t &sr3 = sv3, sv5;
  // CHECK: dpct::event_ptr &sr4 = sr3, *sp4;
  cudaEvent_t &sr4 = sr3, *sp4;
  // CHECK: dpct::event_ptr *sp5, &sr5 = sr4;
  cudaEvent_t *sp5, &sr5 = sr4;
  // CHECK: dpct::event_ptr sv6, *sp6, &sr6 = sr5;
  cudaEvent_t sv6, *sp6, &sr6 = sr5;

  // CHECK: dpct::event_ptr eventArray[23];
  cudaEvent_t eventArray[23];
  // CHECK: std::vector<dpct::event_ptr> eventVector;
  std::vector<cudaEvent_t> eventVector;

  // CHECK: dpct::queue_ptr stream, stream0;
  // CHECK-NEXT: dpct::queue_ptr streams[23], streams0[45];
  // CHECK-NEXT: dpct::queue_ptr streams1[10][10][10], streams2[2][2][2][2];
  // CHECK-NEXT: dpct::queue_ptr *streams3[10][10][10], streams4[2][2][2][2];
  // CHECK-NEXT: dpct::event_ptr event, events[23];
  // CHECK-NEXT: C<dpct::queue_ptr, dpct::event_ptr> se, se2;
  // CHECK-NEXT: std::list<dpct::queue_ptr> streamlist;
  // CHECK-NEXT: std::list<dpct::event_ptr> eventlist;
  // CHECK-NEXT: std::list<int> errors;
  // CHECK-NEXT: std::list<dpct::device_info> props;
  cudaStream_t stream, stream0;
  cudaStream_t streams[23], streams0[45];
  cudaStream_t streams1[10][10][10], streams2[2][2][2][2];
  cudaStream_t *streams3[10][10][10], streams4[2][2][2][2];
  cudaEvent_t event, events[23];
  C<cudaStream_t, cudaEvent_t> se, se2;
  std::list<cudaStream_t> streamlist;
  std::list<cudaEvent_t> eventlist;
  std::list<cudaError> errors;
  std::list<cudaDeviceProp> props;
};

int main(int argc, char* argv[]) {
  // CHECK: dpct::event_ptr v;
  cudaEvent_t v;
  // CHECK: dpct::event_ptr *p;
  cudaEvent_t *p;
  // CHECK: dpct::event_ptr **p1;
  cudaEvent_t **p1;
  // CHECK: dpct::event_ptr *p2 = &v;
  cudaEvent_t *p2 = &v;
  // CHECK: dpct::event_ptr **p3 = &p2;
  cudaEvent_t **p3 = &p2;
  // CHECK: dpct::event_ptr &r = v;
  cudaEvent_t &r = v;

  // CHECK: dpct::event_ptr vv, vv2;
  cudaEvent_t vv, vv2;
  // CHECK: dpct::event_ptr vv3, *pp;
  cudaEvent_t vv3, *pp;
  // CHECK: dpct::event_ptr *pp2, vv4;
  cudaEvent_t *pp2, vv4;
  // CHECK: dpct::event_ptr *pp3, *pp4;
  cudaEvent_t *pp3, *pp4;
  // CHECK: dpct::event_ptr vv5, &rr = vv;
  cudaEvent_t vv5, &rr = vv;
  // CHECK: dpct::event_ptr &rr2 = vv2, vv6;
  cudaEvent_t &rr2 = vv2, vv6;
  // CHECK: dpct::event_ptr *pp5, &rr3 = *pp;
  cudaEvent_t *pp5, &rr3 = *pp;
  // CHECK: dpct::event_ptr &rr4 = *pp5, *pp6 = &vv3;
  cudaEvent_t &rr4 = *pp5, *pp6 = &vv3;

  // CHECK: dpct::event_ptr vvv, vvv2, vvv3;
  cudaEvent_t vvv, vvv2, vvv3;
  // CHECK: dpct::event_ptr *ppp, *ppp2, *ppp3;
  cudaEvent_t *ppp, *ppp2, *ppp3;
  // CHECK: dpct::event_ptr &rrr = vvv, &rrr2 = vvv2, &rrr3 = vvv3;
  cudaEvent_t &rrr = vvv, &rrr2 = vvv2, &rrr3 = vvv3;
  // CHECK: dpct::event_ptr vvv4, *ppp4, &rrr4 = vvv;
  cudaEvent_t vvv4, *ppp4, &rrr4 = vvv;
  // CHECK: dpct::event_ptr *ppp5, vvv5, &rrr5 = vvv;
  cudaEvent_t *ppp5, vvv5, &rrr5 = vvv;
  // CHECK: dpct::event_ptr &rrr6 = vvv, vvv6, *ppp6;
  cudaEvent_t &rrr6 = vvv, vvv6, *ppp6;

  // CHECK: dpct::queue_ptr stream, stream0;
  // CHECK-NEXT: dpct::queue_ptr streams[23], streams0[45];
  // CHECK-NEXT: dpct::event_ptr event, events[23];
  // CHECK-NEXT: C<dpct::queue_ptr, dpct::event_ptr> se, se2;
  // CHECK-NEXT: std::list<dpct::queue_ptr> streamlist;
  // CHECK-NEXT: std::list<dpct::event_ptr> eventlist;
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

