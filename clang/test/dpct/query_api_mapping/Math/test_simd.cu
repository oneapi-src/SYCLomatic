/// SIMD Intrinsics

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabs2 | FileCheck %s -check-prefix=VABS2
// VABS2: CUDA API:
// VABS2-NEXT:   __vabs2(u /*unsigned int*/);
// VABS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABS2-NEXT:   sycl::ext::intel::math::vabs2<unsigned>(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabs4 | FileCheck %s -check-prefix=VABS4
// VABS4: CUDA API:
// VABS4-NEXT:   __vabs4(u /*unsigned int*/);
// VABS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABS4-NEXT:   sycl::ext::intel::math::vabs4<unsigned>(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabsdiffs2 | FileCheck %s -check-prefix=VABSDIFFS2
// VABSDIFFS2: CUDA API:
// VABSDIFFS2-NEXT:   __vabsdiffs2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VABSDIFFS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABSDIFFS2-NEXT:   sycl::ext::intel::math::vabsdiffs2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabsdiffs4 | FileCheck %s -check-prefix=VABSDIFFS4
// VABSDIFFS4: CUDA API:
// VABSDIFFS4-NEXT:   __vabsdiffs4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VABSDIFFS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABSDIFFS4-NEXT:   sycl::ext::intel::math::vabsdiffs4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabsdiffu2 | FileCheck %s -check-prefix=VABSDIFFU2
// VABSDIFFU2: CUDA API:
// VABSDIFFU2-NEXT:   __vabsdiffu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VABSDIFFU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABSDIFFU2-NEXT:   sycl::ext::intel::math::vabsdiffu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabsdiffu4 | FileCheck %s -check-prefix=VABSDIFFU4
// VABSDIFFU4: CUDA API:
// VABSDIFFU4-NEXT:   __vabsdiffu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VABSDIFFU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABSDIFFU4-NEXT:   sycl::ext::intel::math::vabsdiffu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabsss2 | FileCheck %s -check-prefix=VABSSS2
// VABSSS2: CUDA API:
// VABSSS2-NEXT:   __vabsss2(u /*unsigned int*/);
// VABSSS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABSSS2-NEXT:   sycl::ext::intel::math::vabsss2<unsigned>(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vabsss4 | FileCheck %s -check-prefix=VABSSS4
// VABSSS4: CUDA API:
// VABSSS4-NEXT:   __vabsss4(u /*unsigned int*/);
// VABSSS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VABSSS4-NEXT:   sycl::ext::intel::math::vabsss4<unsigned>(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vadd2 | FileCheck %s -check-prefix=VADD2
// VADD2: CUDA API:
// VADD2-NEXT:   __vadd2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VADD2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VADD2-NEXT:   sycl::ext::intel::math::vadd2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vadd4 | FileCheck %s -check-prefix=VADD4
// VADD4: CUDA API:
// VADD4-NEXT:   __vadd4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VADD4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VADD4-NEXT:   sycl::ext::intel::math::vadd4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vaddss2 | FileCheck %s -check-prefix=VADDSS2
// VADDSS2: CUDA API:
// VADDSS2-NEXT:   __vaddss2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VADDSS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VADDSS2-NEXT:   sycl::ext::intel::math::vaddss2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vaddss4 | FileCheck %s -check-prefix=VADDSS4
// VADDSS4: CUDA API:
// VADDSS4-NEXT:   __vaddss4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VADDSS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VADDSS4-NEXT:   sycl::ext::intel::math::vaddss4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vaddus2 | FileCheck %s -check-prefix=VADDUS2
// VADDUS2: CUDA API:
// VADDUS2-NEXT:   __vaddus2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VADDUS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VADDUS2-NEXT:   sycl::ext::intel::math::vaddus2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vaddus4 | FileCheck %s -check-prefix=VADDUS4
// VADDUS4: CUDA API:
// VADDUS4-NEXT:   __vaddus4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VADDUS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VADDUS4-NEXT:   sycl::ext::intel::math::vaddus4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vavgs2 | FileCheck %s -check-prefix=VAVGS2
// VAVGS2: CUDA API:
// VAVGS2-NEXT:   __vavgs2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VAVGS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VAVGS2-NEXT:   sycl::ext::intel::math::vavgs2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vavgs4 | FileCheck %s -check-prefix=VAVGS4
// VAVGS4: CUDA API:
// VAVGS4-NEXT:   __vavgs4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VAVGS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VAVGS4-NEXT:   sycl::ext::intel::math::vavgs4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vavgu2 | FileCheck %s -check-prefix=VAVGU2
// VAVGU2: CUDA API:
// VAVGU2-NEXT:   __vavgu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VAVGU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VAVGU2-NEXT:   sycl::ext::intel::math::vavgu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vavgu4 | FileCheck %s -check-prefix=VAVGU4
// VAVGU4: CUDA API:
// VAVGU4-NEXT:   __vavgu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VAVGU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VAVGU4-NEXT:   sycl::ext::intel::math::vavgu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpeq2 | FileCheck %s -check-prefix=VCMPEQ2
// VCMPEQ2: CUDA API:
// VCMPEQ2-NEXT:   __vcmpeq2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPEQ2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPEQ2-NEXT:   sycl::ext::intel::math::vcmpeq2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpeq4 | FileCheck %s -check-prefix=VCMPEQ4
// VCMPEQ4: CUDA API:
// VCMPEQ4-NEXT:   __vcmpeq4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPEQ4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPEQ4-NEXT:   sycl::ext::intel::math::vcmpeq4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpges2 | FileCheck %s -check-prefix=VCMPGES2
// VCMPGES2: CUDA API:
// VCMPGES2-NEXT:   __vcmpges2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGES2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGES2-NEXT:   sycl::ext::intel::math::vcmpges2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpges4 | FileCheck %s -check-prefix=VCMPGES4
// VCMPGES4: CUDA API:
// VCMPGES4-NEXT:   __vcmpges4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGES4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGES4-NEXT:   sycl::ext::intel::math::vcmpges4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpgeu2 | FileCheck %s -check-prefix=VCMPGEU2
// VCMPGEU2: CUDA API:
// VCMPGEU2-NEXT:   __vcmpgeu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGEU2-NEXT:   sycl::ext::intel::math::vcmpgeu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpgeu4 | FileCheck %s -check-prefix=VCMPGEU4
// VCMPGEU4: CUDA API:
// VCMPGEU4-NEXT:   __vcmpgeu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGEU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGEU4-NEXT:   sycl::ext::intel::math::vcmpgeu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpgts2 | FileCheck %s -check-prefix=VCMPGTS2
// VCMPGTS2: CUDA API:
// VCMPGTS2-NEXT:   __vcmpgts2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGTS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGTS2-NEXT:   sycl::ext::intel::math::vcmpgts2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpgts4 | FileCheck %s -check-prefix=VCMPGTS4
// VCMPGTS4: CUDA API:
// VCMPGTS4-NEXT:   __vcmpgts4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGTS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGTS4-NEXT:   sycl::ext::intel::math::vcmpgts4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpgtu2 | FileCheck %s -check-prefix=VCMPGTU2
// VCMPGTU2: CUDA API:
// VCMPGTU2-NEXT:   __vcmpgtu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGTU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGTU2-NEXT:   sycl::ext::intel::math::vcmpgtu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpgtu4 | FileCheck %s -check-prefix=VCMPGTU4
// VCMPGTU4: CUDA API:
// VCMPGTU4-NEXT:   __vcmpgtu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPGTU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPGTU4-NEXT:   sycl::ext::intel::math::vcmpgtu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmples2 | FileCheck %s -check-prefix=VCMPLES2
// VCMPLES2: CUDA API:
// VCMPLES2-NEXT:   __vcmples2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLES2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLES2-NEXT:   sycl::ext::intel::math::vcmples2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmples4 | FileCheck %s -check-prefix=VCMPLES4
// VCMPLES4: CUDA API:
// VCMPLES4-NEXT:   __vcmples4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLES4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLES4-NEXT:   sycl::ext::intel::math::vcmples4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpleu2 | FileCheck %s -check-prefix=VCMPLEU2
// VCMPLEU2: CUDA API:
// VCMPLEU2-NEXT:   __vcmpleu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLEU2-NEXT:   sycl::ext::intel::math::vcmpleu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpleu4 | FileCheck %s -check-prefix=VCMPLEU4
// VCMPLEU4: CUDA API:
// VCMPLEU4-NEXT:   __vcmpleu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLEU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLEU4-NEXT:   sycl::ext::intel::math::vcmpleu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmplts2 | FileCheck %s -check-prefix=VCMPLTS2
// VCMPLTS2: CUDA API:
// VCMPLTS2-NEXT:   __vcmplts2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLTS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLTS2-NEXT:   sycl::ext::intel::math::vcmplts2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmplts4 | FileCheck %s -check-prefix=VCMPLTS4
// VCMPLTS4: CUDA API:
// VCMPLTS4-NEXT:   __vcmplts4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLTS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLTS4-NEXT:   sycl::ext::intel::math::vcmplts4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpltu2 | FileCheck %s -check-prefix=VCMPLTU2
// VCMPLTU2: CUDA API:
// VCMPLTU2-NEXT:   __vcmpltu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLTU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLTU2-NEXT:   sycl::ext::intel::math::vcmpltu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpltu4 | FileCheck %s -check-prefix=VCMPLTU4
// VCMPLTU4: CUDA API:
// VCMPLTU4-NEXT:   __vcmpltu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPLTU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPLTU4-NEXT:   sycl::ext::intel::math::vcmpltu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpne2 | FileCheck %s -check-prefix=VCMPNE2
// VCMPNE2: CUDA API:
// VCMPNE2-NEXT:   __vcmpne2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPNE2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPNE2-NEXT:   sycl::ext::intel::math::vcmpne2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vcmpne4 | FileCheck %s -check-prefix=VCMPNE4
// VCMPNE4: CUDA API:
// VCMPNE4-NEXT:   __vcmpne4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VCMPNE4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VCMPNE4-NEXT:   sycl::ext::intel::math::vcmpne4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vhaddu2 | FileCheck %s -check-prefix=VHADDU2
// VHADDU2: CUDA API:
// VHADDU2-NEXT:   __vhaddu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VHADDU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VHADDU2-NEXT:   sycl::ext::intel::math::vhaddu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vhaddu4 | FileCheck %s -check-prefix=VHADDU4
// VHADDU4: CUDA API:
// VHADDU4-NEXT:   __vhaddu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VHADDU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VHADDU4-NEXT:   sycl::ext::intel::math::vhaddu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vmaxs2 | FileCheck %s -check-prefix=VMAXS2
// VMAXS2: CUDA API:
// VMAXS2-NEXT:   __vmaxs2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMAXS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMAXS2-NEXT:   sycl::ext::intel::math::vmaxs2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vmaxs4 | FileCheck %s -check-prefix=VMAXS4
// VMAXS4: CUDA API:
// VMAXS4-NEXT:   __vmaxs4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMAXS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMAXS4-NEXT:   sycl::ext::intel::math::vmaxs4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vmaxu2 | FileCheck %s -check-prefix=VMAXU2
// VMAXU2: CUDA API:
// VMAXU2-NEXT:   __vmaxu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMAXU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMAXU2-NEXT:   sycl::ext::intel::math::vmaxu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vmaxu4 | FileCheck %s -check-prefix=VMAXU4
// VMAXU4: CUDA API:
// VMAXU4-NEXT:   __vmaxu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMAXU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMAXU4-NEXT:   sycl::ext::intel::math::vmaxu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vmins2 | FileCheck %s -check-prefix=VMINS2
// VMINS2: CUDA API:
// VMINS2-NEXT:   __vmins2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMINS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMINS2-NEXT:   sycl::ext::intel::math::vmins2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vmins4 | FileCheck %s -check-prefix=VMINS4
// VMINS4: CUDA API:
// VMINS4-NEXT:   __vmins4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMINS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMINS4-NEXT:   sycl::ext::intel::math::vmins4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vminu2 | FileCheck %s -check-prefix=VMINU2
// VMINU2: CUDA API:
// VMINU2-NEXT:   __vminu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMINU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMINU2-NEXT:   sycl::ext::intel::math::vminu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vminu4 | FileCheck %s -check-prefix=VMINU4
// VMINU4: CUDA API:
// VMINU4-NEXT:   __vminu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VMINU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VMINU4-NEXT:   sycl::ext::intel::math::vminu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vneg2 | FileCheck %s -check-prefix=VNEG2
// VNEG2: CUDA API:
// VNEG2-NEXT:   __vneg2(u /*unsigned int*/);
// VNEG2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VNEG2-NEXT:   sycl::ext::intel::math::vneg2<unsigned>(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vneg4 | FileCheck %s -check-prefix=VNEG4
// VNEG4: CUDA API:
// VNEG4-NEXT:   __vneg4(u /*unsigned int*/);
// VNEG4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VNEG4-NEXT:   sycl::ext::intel::math::vneg4<unsigned>(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vnegss2 | FileCheck %s -check-prefix=VNEGSS2
// VNEGSS2: CUDA API:
// VNEGSS2-NEXT:   __vnegss2(u /*unsigned int*/);
// VNEGSS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VNEGSS2-NEXT:   sycl::ext::intel::math::vnegss2<unsigned>(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vnegss4 | FileCheck %s -check-prefix=VNEGSS4
// VNEGSS4: CUDA API:
// VNEGSS4-NEXT:   __vnegss4(u /*unsigned int*/);
// VNEGSS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VNEGSS4-NEXT:   sycl::ext::intel::math::vnegss4<unsigned>(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsads2 | FileCheck %s -check-prefix=VSADS2
// VSADS2: CUDA API:
// VSADS2-NEXT:   __vsads2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSADS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSADS2-NEXT:   sycl::ext::intel::math::vsads2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsads4 | FileCheck %s -check-prefix=VSADS4
// VSADS4: CUDA API:
// VSADS4-NEXT:   __vsads4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSADS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSADS4-NEXT:   sycl::ext::intel::math::vsads4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsadu2 | FileCheck %s -check-prefix=VSADU2
// VSADU2: CUDA API:
// VSADU2-NEXT:   __vsadu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSADU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSADU2-NEXT:   sycl::ext::intel::math::vsadu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsadu4 | FileCheck %s -check-prefix=VSADU4
// VSADU4: CUDA API:
// VSADU4-NEXT:   __vsadu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSADU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSADU4-NEXT:   sycl::ext::intel::math::vsadu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vseteq2 | FileCheck %s -check-prefix=VSETEQ2
// VSETEQ2: CUDA API:
// VSETEQ2-NEXT:   __vseteq2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETEQ2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETEQ2-NEXT:   sycl::ext::intel::math::vseteq2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vseteq4 | FileCheck %s -check-prefix=VSETEQ4
// VSETEQ4: CUDA API:
// VSETEQ4-NEXT:   __vseteq4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETEQ4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETEQ4-NEXT:   sycl::ext::intel::math::vseteq4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetges2 | FileCheck %s -check-prefix=VSETGES2
// VSETGES2: CUDA API:
// VSETGES2-NEXT:   __vsetges2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGES2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGES2-NEXT:   sycl::ext::intel::math::vsetges2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetges4 | FileCheck %s -check-prefix=VSETGES4
// VSETGES4: CUDA API:
// VSETGES4-NEXT:   __vsetges4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGES4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGES4-NEXT:   sycl::ext::intel::math::vsetges4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetgeu2 | FileCheck %s -check-prefix=VSETGEU2
// VSETGEU2: CUDA API:
// VSETGEU2-NEXT:   __vsetgeu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGEU2-NEXT:   sycl::ext::intel::math::vsetgeu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetgeu4 | FileCheck %s -check-prefix=VSETGEU4
// VSETGEU4: CUDA API:
// VSETGEU4-NEXT:   __vsetgeu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGEU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGEU4-NEXT:   sycl::ext::intel::math::vsetgeu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetgts2 | FileCheck %s -check-prefix=VSETGTS2
// VSETGTS2: CUDA API:
// VSETGTS2-NEXT:   __vsetgts2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGTS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGTS2-NEXT:   sycl::ext::intel::math::vsetgts2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetgts4 | FileCheck %s -check-prefix=VSETGTS4
// VSETGTS4: CUDA API:
// VSETGTS4-NEXT:   __vsetgts4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGTS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGTS4-NEXT:   sycl::ext::intel::math::vsetgts4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetgtu2 | FileCheck %s -check-prefix=VSETGTU2
// VSETGTU2: CUDA API:
// VSETGTU2-NEXT:   __vsetgtu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGTU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGTU2-NEXT:   sycl::ext::intel::math::vsetgtu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetgtu4 | FileCheck %s -check-prefix=VSETGTU4
// VSETGTU4: CUDA API:
// VSETGTU4-NEXT:   __vsetgtu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETGTU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETGTU4-NEXT:   sycl::ext::intel::math::vsetgtu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetles2 | FileCheck %s -check-prefix=VSETLES2
// VSETLES2: CUDA API:
// VSETLES2-NEXT:   __vsetles2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLES2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLES2-NEXT:   sycl::ext::intel::math::vsetles2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetles4 | FileCheck %s -check-prefix=VSETLES4
// VSETLES4: CUDA API:
// VSETLES4-NEXT:   __vsetles4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLES4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLES4-NEXT:   sycl::ext::intel::math::vsetles4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetleu2 | FileCheck %s -check-prefix=VSETLEU2
// VSETLEU2: CUDA API:
// VSETLEU2-NEXT:   __vsetleu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLEU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLEU2-NEXT:   sycl::ext::intel::math::vsetleu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetleu4 | FileCheck %s -check-prefix=VSETLEU4
// VSETLEU4: CUDA API:
// VSETLEU4-NEXT:   __vsetleu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLEU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLEU4-NEXT:   sycl::ext::intel::math::vsetleu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetlts2 | FileCheck %s -check-prefix=VSETLTS2
// VSETLTS2: CUDA API:
// VSETLTS2-NEXT:   __vsetlts2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLTS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLTS2-NEXT:   sycl::ext::intel::math::vsetlts2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetlts4 | FileCheck %s -check-prefix=VSETLTS4
// VSETLTS4: CUDA API:
// VSETLTS4-NEXT:   __vsetlts4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLTS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLTS4-NEXT:   sycl::ext::intel::math::vsetlts4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetltu2 | FileCheck %s -check-prefix=VSETLTU2
// VSETLTU2: CUDA API:
// VSETLTU2-NEXT:   __vsetltu2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLTU2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLTU2-NEXT:   sycl::ext::intel::math::vsetltu2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetltu4 | FileCheck %s -check-prefix=VSETLTU4
// VSETLTU4: CUDA API:
// VSETLTU4-NEXT:   __vsetltu4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETLTU4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETLTU4-NEXT:   sycl::ext::intel::math::vsetltu4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetne2 | FileCheck %s -check-prefix=VSETNE2
// VSETNE2: CUDA API:
// VSETNE2-NEXT:   __vsetne2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETNE2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETNE2-NEXT:   sycl::ext::intel::math::vsetne2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsetne4 | FileCheck %s -check-prefix=VSETNE4
// VSETNE4: CUDA API:
// VSETNE4-NEXT:   __vsetne4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSETNE4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSETNE4-NEXT:   sycl::ext::intel::math::vsetne4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsub2 | FileCheck %s -check-prefix=VSUB2
// VSUB2: CUDA API:
// VSUB2-NEXT:   __vsub2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSUB2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSUB2-NEXT:   sycl::ext::intel::math::vsub2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsub4 | FileCheck %s -check-prefix=VSUB4
// VSUB4: CUDA API:
// VSUB4-NEXT:   __vsub4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSUB4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSUB4-NEXT:   sycl::ext::intel::math::vsub4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsubss2 | FileCheck %s -check-prefix=VSUBSS2
// VSUBSS2: CUDA API:
// VSUBSS2-NEXT:   __vsubss2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSUBSS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSUBSS2-NEXT:   sycl::ext::intel::math::vsubss2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsubss4 | FileCheck %s -check-prefix=VSUBSS4
// VSUBSS4: CUDA API:
// VSUBSS4-NEXT:   __vsubss4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSUBSS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSUBSS4-NEXT:   sycl::ext::intel::math::vsubss4<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsubus2 | FileCheck %s -check-prefix=VSUBUS2
// VSUBUS2: CUDA API:
// VSUBUS2-NEXT:   __vsubus2(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSUBUS2-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSUBUS2-NEXT:   sycl::ext::intel::math::vsubus2<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__vsubus4 | FileCheck %s -check-prefix=VSUBUS4
// VSUBUS4: CUDA API:
// VSUBUS4-NEXT:   __vsubus4(u1 /*unsigned int*/, u2 /*unsigned int*/);
// VSUBUS4-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// VSUBUS4-NEXT:   sycl::ext::intel::math::vsubus4<unsigned>(u1, u2);
