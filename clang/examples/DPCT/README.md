# How to add API mapping cases

- Add the API mapping cases of different libraries in corresponding folder.

- Make sure the file name is format as `${API_Name}.cu` (change the `:` to `$`).

- Content of `${API_Name}.cu`, take the `__hfma.cu` as an example (The comment with "///" is not the content of the file):

```c++
/// If the case need some option, add them at the first line started with "// Option:".
// Option: --use-dpcpp-extensions=intel_device_math
#include "cuda_bf16.h"
#include "cuda_fp16.h"

/// Just have one function definition.
__global__ void test(__half h1, __half h2, __half h3, __nv_bfloat16 b1,
                     __nv_bfloat16 b2, __nv_bfloat16 b3) {
/// Put the code that need show to users between the "// Start" line and the "// End" line.
  // Start
/// Add the type after each argument, and the argument name can be simple.
  __hfma(h1 /*__half*/, h2 /*__half*/, h3 /*__half*/);
  __hfma(b1 /*__nv_bfloat16*/, b2 /*__nv_bfloat16*/, b3 /*__nv_bfloat16*/);
  // End
}
```

- If the case need not be migrate, add the output message at the first line started with `// Migration desc: `.

- Please format the code.

- Add each API mapping test case in corresponding folder of `test/dpct/query_api_mapping`.

- If the API has more than 5 overload functions, add the index number (like: `/* 1 */`) before each function.
