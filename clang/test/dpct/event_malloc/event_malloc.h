class C {
public:
  C();  // constructor declaration
  ~C(); // destructor declaration

private:
  cudaEvent_t *kernelEvent;
};

// CHECK: C::C(void) { kernelEvent = (dpct::event_ptr *)malloc(4 * sizeof(dpct::event_ptr)); }
C::C(void) { kernelEvent = (cudaEvent_t *)malloc(4 * sizeof(cudaEvent_t)); }
