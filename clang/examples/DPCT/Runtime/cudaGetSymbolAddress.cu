#define MAX_CONST_SIZE 1024
__constant__ char symbol[MAX_CONST_SIZE];

void test(void **pDev) {
  // Start
  cudaGetSymbolAddress(pDev /*void ***/, symbol /*const void **/);
  // End
}
