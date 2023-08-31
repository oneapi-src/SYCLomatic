void test(cudaPointerAttributes *attr) {
  // Start
  const void *ptr;
  cudaPointerGetAttributes(attr /*cudaPointerAttributes **/,
                           ptr /*const void **/);
  // End
}
