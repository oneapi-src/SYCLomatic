// RUN: dpct --format-range=none -out-root %T/output  %S/test2.cu %s --cuda-include-path="%cuda-path/include" -- -x cuda -I %S/../

// When the macro spelling location did not include in the --in-root path, the dpct will crash with SIGABRT.
#include "test.h"
#include <cuda.h>
void test(const char* value) {
    TEST_DCHECK(value != NULL);
}