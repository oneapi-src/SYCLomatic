/// Double Precision Intrinsics

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dadd_rd | FileCheck %s -check-prefix=__DADD_RD
// __DADD_RD: CUDA API:
// __DADD_RD-NEXT:   __dadd_rd(d1 /*double*/, d2 /*double*/);
// __DADD_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DADD_RD-NEXT:   sycl::ext::intel::math::dadd_rd(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dadd_rn | FileCheck %s -check-prefix=__DADD_RN
// __DADD_RN: CUDA API:
// __DADD_RN-NEXT:   __dadd_rn(d1 /*double*/, d2 /*double*/);
// __DADD_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DADD_RN-NEXT:   sycl::ext::intel::math::dadd_rn(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dadd_ru | FileCheck %s -check-prefix=__DADD_RU
// __DADD_RU: CUDA API:
// __DADD_RU-NEXT:   __dadd_ru(d1 /*double*/, d2 /*double*/);
// __DADD_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DADD_RU-NEXT:   sycl::ext::intel::math::dadd_ru(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dadd_rz | FileCheck %s -check-prefix=__DADD_RZ
// __DADD_RZ: CUDA API:
// __DADD_RZ-NEXT:   __dadd_rz(d1 /*double*/, d2 /*double*/);
// __DADD_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DADD_RZ-NEXT:   sycl::ext::intel::math::dadd_rz(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ddiv_rd | FileCheck %s -check-prefix=__DDIV_RD
// __DDIV_RD: CUDA API:
// __DDIV_RD-NEXT:   __ddiv_rd(d1 /*double*/, d2 /*double*/);
// __DDIV_RD-NEXT: Is migrated to:
// __DDIV_RD-NEXT:   d1 / d2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ddiv_rn | FileCheck %s -check-prefix=__DDIV_RN
// __DDIV_RN: CUDA API:
// __DDIV_RN-NEXT:   __ddiv_rn(d1 /*double*/, d2 /*double*/);
// __DDIV_RN-NEXT: Is migrated to:
// __DDIV_RN-NEXT:   d1 / d2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ddiv_ru | FileCheck %s -check-prefix=__DDIV_RU
// __DDIV_RU: CUDA API:
// __DDIV_RU-NEXT:   __ddiv_ru(d1 /*double*/, d2 /*double*/);
// __DDIV_RU-NEXT: Is migrated to:
// __DDIV_RU-NEXT:   d1 / d2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ddiv_rz | FileCheck %s -check-prefix=__DDIV_RZ
// __DDIV_RZ: CUDA API:
// __DDIV_RZ-NEXT:   __ddiv_rz(d1 /*double*/, d2 /*double*/);
// __DDIV_RZ-NEXT: Is migrated to:
// __DDIV_RZ-NEXT:   d1 / d2;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dmul_rd | FileCheck %s -check-prefix=__DMUL_RD
// __DMUL_RD: CUDA API:
// __DMUL_RD-NEXT:   __dmul_rd(d1 /*double*/, d2 /*double*/);
// __DMUL_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DMUL_RD-NEXT:   sycl::ext::intel::math::dmul_rd(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dmul_rn | FileCheck %s -check-prefix=__DMUL_RN
// __DMUL_RN: CUDA API:
// __DMUL_RN-NEXT:   __dmul_rn(d1 /*double*/, d2 /*double*/);
// __DMUL_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DMUL_RN-NEXT:   sycl::ext::intel::math::dmul_rn(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dmul_ru | FileCheck %s -check-prefix=__DMUL_RU
// __DMUL_RU: CUDA API:
// __DMUL_RU-NEXT:   __dmul_ru(d1 /*double*/, d2 /*double*/);
// __DMUL_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DMUL_RU-NEXT:   sycl::ext::intel::math::dmul_ru(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dmul_rz | FileCheck %s -check-prefix=__DMUL_RZ
// __DMUL_RZ: CUDA API:
// __DMUL_RZ-NEXT:   __dmul_rz(d1 /*double*/, d2 /*double*/);
// __DMUL_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DMUL_RZ-NEXT:   sycl::ext::intel::math::dmul_rz(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__drcp_rd | FileCheck %s -check-prefix=__DRCP_RD
// __DRCP_RD: CUDA API:
// __DRCP_RD-NEXT:   __drcp_rd(d /*double*/);
// __DRCP_RD-NEXT: Is migrated to:
// __DRCP_RD-NEXT:   (1.0/d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__drcp_rn | FileCheck %s -check-prefix=__DRCP_RN
// __DRCP_RN: CUDA API:
// __DRCP_RN-NEXT:   __drcp_rn(d /*double*/);
// __DRCP_RN-NEXT: Is migrated to:
// __DRCP_RN-NEXT:   (1.0/d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__drcp_ru | FileCheck %s -check-prefix=__DRCP_RU
// __DRCP_RU: CUDA API:
// __DRCP_RU-NEXT:   __drcp_ru(d /*double*/);
// __DRCP_RU-NEXT: Is migrated to:
// __DRCP_RU-NEXT:   (1.0/d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__drcp_rz | FileCheck %s -check-prefix=__DRCP_RZ
// __DRCP_RZ: CUDA API:
// __DRCP_RZ-NEXT:   __drcp_rz(d /*double*/);
// __DRCP_RZ-NEXT: Is migrated to:
// __DRCP_RZ-NEXT:   (1.0/d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dsqrt_rd | FileCheck %s -check-prefix=__DSQRT_RD
// __DSQRT_RD: CUDA API:
// __DSQRT_RD-NEXT:   __dsqrt_rd(d /*double*/);
// __DSQRT_RD-NEXT: Is migrated to:
// __DSQRT_RD-NEXT:   sycl::sqrt(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dsqrt_rn | FileCheck %s -check-prefix=__DSQRT_RN
// __DSQRT_RN: CUDA API:
// __DSQRT_RN-NEXT:   __dsqrt_rn(d /*double*/);
// __DSQRT_RN-NEXT: Is migrated to:
// __DSQRT_RN-NEXT:   sycl::sqrt(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dsqrt_ru | FileCheck %s -check-prefix=__DSQRT_RU
// __DSQRT_RU: CUDA API:
// __DSQRT_RU-NEXT:   __dsqrt_ru(d /*double*/);
// __DSQRT_RU-NEXT: Is migrated to:
// __DSQRT_RU-NEXT:   sycl::sqrt(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dsqrt_rz | FileCheck %s -check-prefix=__DSQRT_RZ
// __DSQRT_RZ: CUDA API:
// __DSQRT_RZ-NEXT:   __dsqrt_rz(d /*double*/);
// __DSQRT_RZ-NEXT: Is migrated to:
// __DSQRT_RZ-NEXT:   sycl::sqrt(d);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dsub_rd | FileCheck %s -check-prefix=__DSUB_RD
// __DSUB_RD: CUDA API:
// __DSUB_RD-NEXT:   __dsub_rd(d1 /*double*/, d2 /*double*/);
// __DSUB_RD-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DSUB_RD-NEXT:   sycl::ext::intel::math::dsub_rd(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dsub_rn | FileCheck %s -check-prefix=__DSUB_RN
// __DSUB_RN: CUDA API:
// __DSUB_RN-NEXT:   __dsub_rn(d1 /*double*/, d2 /*double*/);
// __DSUB_RN-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DSUB_RN-NEXT:   sycl::ext::intel::math::dsub_rn(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dsub_ru | FileCheck %s -check-prefix=__DSUB_RU
// __DSUB_RU: CUDA API:
// __DSUB_RU-NEXT:   __dsub_ru(d1 /*double*/, d2 /*double*/);
// __DSUB_RU-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DSUB_RU-NEXT:   sycl::ext::intel::math::dsub_ru(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dsub_rz | FileCheck %s -check-prefix=__DSUB_RZ
// __DSUB_RZ: CUDA API:
// __DSUB_RZ-NEXT:   __dsub_rz(d1 /*double*/, d2 /*double*/);
// __DSUB_RZ-NEXT: Is migrated to (with the option --use-dpcpp-extensions=intel_device_math):
// __DSUB_RZ-NEXT:   sycl::ext::intel::math::dsub_rz(d1, d2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fma_rd | FileCheck %s -check-prefix=__FMA_RD
// __FMA_RD: CUDA API:
// __FMA_RD-NEXT:   __fma_rd(d1 /*double*/, d2 /*double*/, d3 /*double*/);
// __FMA_RD-NEXT: Is migrated to:
// __FMA_RD-NEXT:   sycl::fma(d1, d2, d3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fma_rn | FileCheck %s -check-prefix=__FMA_RN
// __FMA_RN: CUDA API:
// __FMA_RN-NEXT:   __fma_rn(d1 /*double*/, d2 /*double*/, d3 /*double*/);
// __FMA_RN-NEXT: Is migrated to:
// __FMA_RN-NEXT:   sycl::fma(d1, d2, d3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fma_ru | FileCheck %s -check-prefix=__FMA_RU
// __FMA_RU: CUDA API:
// __FMA_RU-NEXT:   __fma_ru(d1 /*double*/, d2 /*double*/, d3 /*double*/);
// __FMA_RU-NEXT: Is migrated to:
// __FMA_RU-NEXT:   sycl::fma(d1, d2, d3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__fma_rz | FileCheck %s -check-prefix=__FMA_RZ
// __FMA_RZ: CUDA API:
// __FMA_RZ-NEXT:   __fma_rz(d1 /*double*/, d2 /*double*/, d3 /*double*/);
// __FMA_RZ-NEXT: Is migrated to:
// __FMA_RZ-NEXT:   sycl::fma(d1, d2, d3);

/// Integer Intrinsics

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__brev | FileCheck %s -check-prefix=__BREV
// __BREV: CUDA API:
// __BREV-NEXT:   __brev(u /*unsigned int*/);
// __BREV-NEXT: Is migrated to:
// __BREV-NEXT:   dpct::reverse_bits<unsigned int>(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__brevll | FileCheck %s -check-prefix=__BREVLL
// __BREVLL: CUDA API:
// __BREVLL-NEXT:   __brevll(ull /*unsigned long long int*/);
// __BREVLL-NEXT: Is migrated to:
// __BREVLL-NEXT:   dpct::reverse_bits<unsigned long long>(ull);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__byte_perm | FileCheck %s -check-prefix=__BYTE_PERM
// __BYTE_PERM: CUDA API:
// __BYTE_PERM-NEXT:   __byte_perm(u1 /*unsigned int*/, u2 /*unsigned int*/, u3 /*unsigned int*/);
// __BYTE_PERM-NEXT: Is migrated to:
// __BYTE_PERM-NEXT:   dpct::byte_level_permute(u1, u2, u3);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__clz | FileCheck %s -check-prefix=__CLZ
// __CLZ: CUDA API:
// __CLZ-NEXT:   __clz(i /*int*/);
// __CLZ-NEXT: Is migrated to:
// __CLZ-NEXT:   sycl::clz(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__clzll | FileCheck %s -check-prefix=__CLZLL
// __CLZLL: CUDA API:
// __CLZLL-NEXT:   __clzll(ll /*long long int*/);
// __CLZLL-NEXT: Is migrated to:
// __CLZLL-NEXT:   sycl::clz(ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ffs | FileCheck %s -check-prefix=__FFS
// __FFS: CUDA API:
// __FFS-NEXT:   __ffs(i /*int*/);
// __FFS-NEXT: Is migrated to:
// __FFS-NEXT:   dpct::ffs<int>(i);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__ffsll | FileCheck %s -check-prefix=__FFSLL
// __FFSLL: CUDA API:
// __FFSLL-NEXT:   __ffsll(ll /*long long int*/);
// __FFSLL-NEXT: Is migrated to:
// __FFSLL-NEXT:   dpct::ffs<long long int>(ll);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__mul24 | FileCheck %s -check-prefix=__MUL24
// __MUL24: CUDA API:
// __MUL24-NEXT:   __mul24(i1 /*int*/, i2 /*int*/);
// __MUL24-NEXT: Is migrated to:
// __MUL24-NEXT:   sycl::mul24(i1, i2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__mul64hi | FileCheck %s -check-prefix=__MUL64HI
// __MUL64HI: CUDA API:
// __MUL64HI-NEXT:   __mul64hi(ll1 /*long long int*/, ll2 /*long long int*/);
// __MUL64HI-NEXT: Is migrated to:
// __MUL64HI-NEXT:   sycl::mul_hi(ll1, ll2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__mulhi | FileCheck %s -check-prefix=__MULHI
// __MULHI: CUDA API:
// __MULHI-NEXT:   __mulhi(i1 /*int*/, i2 /*int*/);
// __MULHI-NEXT: Is migrated to:
// __MULHI-NEXT:   sycl::mul_hi(i1, i2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__popc | FileCheck %s -check-prefix=__POPC
// __POPC: CUDA API:
// __POPC-NEXT:   __popc(u /*unsigned int*/);
// __POPC-NEXT: Is migrated to:
// __POPC-NEXT:   sycl::popcount(u);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__popcll | FileCheck %s -check-prefix=__POPCLL
// __POPCLL: CUDA API:
// __POPCLL-NEXT:   __popcll(ull /*unsigned long long int*/);
// __POPCLL-NEXT: Is migrated to:
// __POPCLL-NEXT:   sycl::popcount(ull);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__rhadd | FileCheck %s -check-prefix=__RHADD
// __RHADD: CUDA API:
// __RHADD-NEXT:   __rhadd(i1 /*int*/, i2 /*int*/);
// __RHADD-NEXT: Is migrated to:
// __RHADD-NEXT:   sycl::rhadd(i1, i2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__sad | FileCheck %s -check-prefix=__SAD
// __SAD: CUDA API:
// __SAD-NEXT:   __sad(i1 /*int*/, i2 /*int*/, u /*unsigned int*/);
// __SAD-NEXT: Is migrated to:
// __SAD-NEXT:   sycl::abs_diff(i1, i2)+u;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__uhadd | FileCheck %s -check-prefix=__UHADD
// __UHADD: CUDA API:
// __UHADD-NEXT:   __uhadd(u1 /*unsigned int*/, u2 /*unsigned int*/);
// __UHADD-NEXT: Is migrated to:
// __UHADD-NEXT:   sycl::hadd(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__umul24 | FileCheck %s -check-prefix=__UMUL24
// __UMUL24: CUDA API:
// __UMUL24-NEXT:   __umul24(u1 /*unsigned int*/, u2 /*unsigned int*/);
// __UMUL24-NEXT: Is migrated to:
// __UMUL24-NEXT:   sycl::mul24(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__umul64hi | FileCheck %s -check-prefix=__UMUL64HI
// __UMUL64HI: CUDA API:
// __UMUL64HI-NEXT:   __umul64hi(ull1 /*unsigned long long int*/, ull2 /*unsigned long long int*/);
// __UMUL64HI-NEXT: Is migrated to:
// __UMUL64HI-NEXT:   sycl::mul_hi(ull1 /*unsigned long long int*/, ull2 /*unsigned long long int*/);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__umulhi | FileCheck %s -check-prefix=__UMULHI
// __UMULHI: CUDA API:
// __UMULHI-NEXT:   __umulhi(u1 /*unsigned int*/, u2 /*unsigned int*/);
// __UMULHI-NEXT: Is migrated to:
// __UMULHI-NEXT:   sycl::mul_hi<unsigned>(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__urhadd | FileCheck %s -check-prefix=__URHADD
// __URHADD: CUDA API:
// __URHADD-NEXT:   __urhadd(u1 /*unsigned int*/, u2 /*unsigned int*/);
// __URHADD-NEXT: Is migrated to:
// __URHADD-NEXT:   sycl::rhadd(u1, u2);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__usad | FileCheck %s -check-prefix=__USAD
// __USAD: CUDA API:
// __USAD-NEXT:   __usad(u1 /*unsigned int*/, u2 /*unsigned int*/, u3 /*unsigned int*/);
// __USAD-NEXT: Is migrated to:
// __USAD-NEXT:   sycl::abs_diff(u1, u2)+u3;

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dp2a_hi | FileCheck %s -check-prefix=__dp2a_hi
// __dp2a_hi: CUDA API:
// __dp2a_hi-NEXT:   __dp2a_hi(i_a /*int*/, i_b /*int*/, i_c /*int*/);
// __dp2a_hi-NEXT:   __dp2a_hi(ui_a /*unsigned int*/, ui_b /*unsigned int*/,
// __dp2a_hi-NEXT:             ui_c /*unsigned int*/);
// __dp2a_hi-NEXT:   __dp2a_hi(us2_a /*ushort2*/, uc4_b /*uchar4*/, ui_c /*unsigned int*/);
// __dp2a_hi-NEXT:   __dp2a_hi(s2_a /*short2*/, c4_b /*char4*/, i_c /*int*/);
// __dp2a_hi-NEXT: Is migrated to:
// __dp2a_hi-NEXT:   dpct::dp2a_hi(i_a, i_b, i_c);
// __dp2a_hi-NEXT:   dpct::dp2a_hi(ui_a, ui_b, ui_c);
// __dp2a_hi-NEXT:   dpct::dp2a_hi(us2_a, uc4_b, ui_c);
// __dp2a_hi-NEXT:   dpct::dp2a_hi(s2_a, c4_b, i_c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dp2a_lo | FileCheck %s -check-prefix=__dp2a_lo
// __dp2a_lo: CUDA API:
// __dp2a_lo-NEXT:   __dp2a_lo(i_a /*int*/, i_b /*int*/, i_c /*int*/);
// __dp2a_lo-NEXT:   __dp2a_lo(ui_a /*unsigned int*/, ui_b /*unsigned int*/,
// __dp2a_lo-NEXT:             ui_c /*unsigned int*/);
// __dp2a_lo-NEXT:   __dp2a_lo(us2_a /*ushort2*/, uc4_b /*uchar4*/, ui_c /*unsigned int*/);
// __dp2a_lo-NEXT:   __dp2a_lo(s2_a /*short2*/, c4_b /*char4*/, i_c /*int*/);
// __dp2a_lo-NEXT: Is migrated to:
// __dp2a_lo-NEXT:   dpct::dp2a_lo(i_a, i_b, i_c);
// __dp2a_lo-NEXT:   dpct::dp2a_lo(ui_a, ui_b, ui_c);
// __dp2a_lo-NEXT:   dpct::dp2a_lo(us2_a, uc4_b, ui_c);
// __dp2a_lo-NEXT:   dpct::dp2a_lo(s2_a, c4_b, i_c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__dp4a | FileCheck %s -check-prefix=__dp4a
// __dp4a: CUDA API:
// __dp4a-NEXT:   __dp4a(i_a /*int*/, i_b /*int*/, i_c /*int*/);
// __dp4a-NEXT:   __dp4a(ui_a /*unsigned int*/, ui_b /*unsigned int*/, ui_c /*unsigned int*/);
// __dp4a-NEXT:   __dp4a(uc4_a /*uchar4*/, uc4_b /*uchar4*/, ui_c /*unsigned int*/);
// __dp4a-NEXT:   __dp4a(c4_a /*char4*/, c4_b /*char4*/, i_c /*int*/);
// __dp4a-NEXT: Is migrated to:
// __dp4a-NEXT:   dpct::dp4a(i_a, i_b, i_c);
// __dp4a-NEXT:   dpct::dp4a(ui_a, ui_b, ui_c);
// __dp4a-NEXT:   dpct::dp4a(uc4_a, uc4_b, ui_c);
// __dp4a-NEXT:   dpct::dp4a(c4_a, c4_b, i_c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__funnelshift_l | FileCheck %s -check-prefix=__funnelshift_l
// __funnelshift_l: CUDA API:
// __funnelshift_l-NEXT:   __funnelshift_l(ui_a /*unsigned int*/, ui_b /*unsigned int*/,
// __funnelshift_l-NEXT:                   ui_c /*unsigned int*/);
// __funnelshift_l-NEXT: Is migrated to:
// __funnelshift_l-NEXT:   dpct::funnelshift_l(ui_a, ui_b, ui_c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__funnelshift_lc | FileCheck %s -check-prefix=__funnelshift_lc
// __funnelshift_lc: CUDA API:
// __funnelshift_lc-NEXT:   __funnelshift_lc(ui_a /*unsigned int*/, ui_b /*unsigned int*/,
// __funnelshift_lc-NEXT:                    ui_c /*unsigned int*/);
// __funnelshift_lc-NEXT: Is migrated to:
// __funnelshift_lc-NEXT:   dpct::funnelshift_lc(ui_a, ui_b, ui_c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__funnelshift_r | FileCheck %s -check-prefix=__funnelshift_r
// __funnelshift_r: CUDA API:
// __funnelshift_r-NEXT:   __funnelshift_r(ui_a /*unsigned int*/, ui_b /*unsigned int*/,
// __funnelshift_r-NEXT:                   ui_c /*unsigned int*/);
// __funnelshift_r-NEXT: Is migrated to:
// __funnelshift_r-NEXT:   dpct::funnelshift_r(ui_a, ui_b, ui_c);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=__funnelshift_rc | FileCheck %s -check-prefix=__funnelshift_rc
// __funnelshift_rc: CUDA API:
// __funnelshift_rc-NEXT:   __funnelshift_rc(ui_a /*unsigned int*/, ui_b /*unsigned int*/,
// __funnelshift_rc-NEXT:                    ui_c /*unsigned int*/);
// __funnelshift_rc-NEXT: Is migrated to:
// __funnelshift_rc-NEXT:   dpct::funnelshift_rc(ui_a, ui_b, ui_c);
