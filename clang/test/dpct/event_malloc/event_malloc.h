class C {
public:
  C();  // constructor declaration
  ~C(); // destructor declaration

private:
  cudaEvent_t *kernelEvent;
};

// CHECK: C::C(void) { kernelEvent = new sycl::event[4]; }
C::C(void) { kernelEvent = (cudaEvent_t *)malloc(4 * sizeof(cudaEvent_t)); }
