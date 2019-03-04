// RUN: syclct -out-root %T %s -- -std=c++11 -x cuda --cuda-host-only --cuda-path=%cuda-path
// RUN: FileCheck --input-file %T/cuda_vardecl.sycl.cpp --match-full-lines %s

// CHECK: cl::sycl::event gv;
cudaEvent_t gv;
// CHECK: cl::sycl::event *gp = &gv;
cudaEvent_t *gp = &gv;
// CHECK: cl::sycl::event &gr = *gp;
cudaEvent_t &gr = *gp;

// CHECK: cl::sycl::event *gp2, gv2;
cudaEvent_t *gp2, gv2;
// CHECK: cl::sycl::event gv3, *gp3;
cudaEvent_t gv3, *gp3;
// CHECK: cl::sycl::event gv4, &gr2 = gv2;
cudaEvent_t gv4, &gr2 = gv2;
// CHECK: cl::sycl::event &gr3 = gv3, gv5;
cudaEvent_t &gr3 = gv3, gv5;
// CHECK: cl::sycl::event &gr4 = gr3, *gp4;
cudaEvent_t &gr4 = gr3, *gp4;
// CHECK: cl::sycl::event *gp5, &gr5 = gr4;
cudaEvent_t *gp5, &gr5 = gr4;
// CHECK: cl::sycl::event gv6, *gp6, &gr6 = gr5;
cudaEvent_t gv6, *gp6, &gr6 = gr5;

// CHECK: void foo(cl::sycl::event paramV, cl::sycl::event *paramP, cl::sycl::event &paramR) try {
void foo(cudaEvent_t paramV, cudaEvent_t *paramP, cudaEvent_t &paramR) {
}

struct S {
  // CHECK: cl::sycl::event sv;
  cudaEvent_t sv;
  // CHECK: cl::sycl::event *sp = &sv;
  cudaEvent_t *sp = &sv;
  // CHECK: cl::sycl::event &sr = *sp;
  cudaEvent_t &sr = *sp;

  // CHECK: cl::sycl::event *sp2, sv2;
  cudaEvent_t *sp2, sv2;
  // CHECK: cl::sycl::event sv3, *sp3;
  cudaEvent_t sv3, *sp3;
  // CHECK: cl::sycl::event sv4, &sr2 = sv2;
  cudaEvent_t sv4, &sr2 = sv2;
  // CHECK: cl::sycl::event &sr3 = sv3, sv5;
  cudaEvent_t &sr3 = sv3, sv5;
  // CHECK: cl::sycl::event &sr4 = sr3, *sp4;
  cudaEvent_t &sr4 = sr3, *sp4;
  // CHECK: cl::sycl::event *sp5, &sr5 = sr4;
  cudaEvent_t *sp5, &sr5 = sr4;
  // CHECK: cl::sycl::event sv6, *sp6, &sr6 = sr5;
  cudaEvent_t sv6, *sp6, &sr6 = sr5;
};

int main(int argc, char* argv[]) {
  // CHECK: cl::sycl::event v;
  cudaEvent_t v;
  // CHECK: cl::sycl::event *p;
  cudaEvent_t *p;
  // CHECK: cl::sycl::event **p1;
  cudaEvent_t **p1;
  // CHECK: cl::sycl::event *p2 = &v;
  cudaEvent_t *p2 = &v;
  // CHECK: cl::sycl::event **p3 = &p2;
  cudaEvent_t **p3 = &p2;
  // CHECK: cl::sycl::event &r = v;
  cudaEvent_t &r = v;

  // CHECK: cl::sycl::event vv, vv2;
  cudaEvent_t vv, vv2;
  // CHECK: cl::sycl::event vv3, *pp;
  cudaEvent_t vv3, *pp;
  // CHECK: cl::sycl::event *pp2, vv4;
  cudaEvent_t *pp2, vv4;
  // CHECK: cl::sycl::event *pp3, *pp4;
  cudaEvent_t *pp3, *pp4;
  // CHECK: cl::sycl::event vv5, &rr = vv;
  cudaEvent_t vv5, &rr = vv;
  // CHECK: cl::sycl::event &rr2 = vv2, vv6;
  cudaEvent_t &rr2 = vv2, vv6;
  // CHECK: cl::sycl::event *pp5, &rr3 = *pp;
  cudaEvent_t *pp5, &rr3 = *pp;
  // CHECK: cl::sycl::event &rr4 = *pp5, *pp6 = &vv3;
  cudaEvent_t &rr4 = *pp5, *pp6 = &vv3;

  // CHECK: cl::sycl::event vvv, vvv2, vvv3;
  cudaEvent_t vvv, vvv2, vvv3;
  // CHECK: cl::sycl::event *ppp, *ppp2, *ppp3;
  cudaEvent_t *ppp, *ppp2, *ppp3;
  // CHECK: cl::sycl::event &rrr = vvv, &rrr2 = vvv2, &rrr3 = vvv3;
  cudaEvent_t &rrr = vvv, &rrr2 = vvv2, &rrr3 = vvv3;
  // CHECK: cl::sycl::event vvv4, *ppp4, &rrr4 = vvv;
  cudaEvent_t vvv4, *ppp4, &rrr4 = vvv;
  // CHECK: cl::sycl::event *ppp5, vvv5, &rrr5 = vvv;
  cudaEvent_t *ppp5, vvv5, &rrr5 = vvv;
  // CHECK: cl::sycl::event &rrr6 = vvv, vvv6, *ppp6;
  cudaEvent_t &rrr6 = vvv, vvv6, *ppp6;
}
