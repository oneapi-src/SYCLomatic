#include "cusparse.h"

void test(cusparseSolveAnalysisInfo_t info) {
  // Start
  cusparseDestroySolveAnalysisInfo(info /*cusparseSolveAnalysisInfo_t*/);
  // End
}
