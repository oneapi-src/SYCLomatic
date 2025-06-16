// UNSUPPORTED: cuda-8.0, cuda-9.0, cuda-9.1, cuda-9.2
// UNSUPPORTED: v8.0, v9.0, v9.1, v9.2

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=llabs | FileCheck %s -check-prefix=llabs
// llabs: CUDA API:
// llabs-NEXT:   llabs(ll /*long long*/);
// llabs-NEXT: Is migrated to:
// llabs-NEXT:   sycl::abs(ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=labs | FileCheck %s -check-prefix=labs
// labs: CUDA API:
// labs-NEXT:   labs(l /*long*/);
// labs-NEXT: Is migrated to:
// labs-NEXT:   sycl::abs(l);
