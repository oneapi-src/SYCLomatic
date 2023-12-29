#define __host__ // May insert these code in dpct binary.
struct size_t {};
struct cudaError_t {};
struct cudaArray_t {};
struct cudaMemcpyKind {};
extern __host__ cudaError_t cudaMemcpy2DArrayToArray(cudaArray_t, size_t,
                                                     size_t, cudaArray_t,
                                                     size_t, size_t, size_t,
                                                     size_t, cudaMemcpyKind);

void test(size_t s1, size_t s2, size_t s3, size_t s4, size_t s5, size_t s6) {
  // TODO: src's type need to be changed to cudaArray_const_t
  // Start
  cudaArray_t dst;
  cudaArray_t src;
  cudaMemcpyKind m;
  cudaMemcpy2DArrayToArray(dst, s1 /*size_t*/, s2 /*size_t*/, src,
                           s3 /*size_t*/, s4 /*size_t*/, s5 /*size_t*/,
                           s6 /*size_t*/, m);
  // End
}
