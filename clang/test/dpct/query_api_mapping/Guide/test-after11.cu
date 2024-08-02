// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2, cuda-10.0, cuda-10.1, cuda-10.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2, v10.0, v10.1, v10.2

/// Nanosleep Function

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__nanosleep | FileCheck %s -check-prefix=__NANOSLEEP
// __NANOSLEEP: CUDA API:
// __NANOSLEEP-NEXT:   __nanosleep(u /*unsigned*/);
// __NANOSLEEP-NEXT: Is migrated to:
// __NANOSLEEP-NEXT:   /*
// __NANOSLEEP-NEXT:   DPCT1008:0: __nanosleep function is not defined in SYCL. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
// __NANOSLEEP-NEXT:   */
// __NANOSLEEP-NEXT:   __nanosleep(u /*unsigned*/);
