// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftExecZ2Z | FileCheck %s -check-prefix=cufftExecZ2Z
// cufftExecZ2Z: CUDA API:
// cufftExecZ2Z-NEXT:   cufftHandle plan;
// cufftExecZ2Z-NEXT:   cufftCreate(&plan /*cufftHandle **/);
// cufftExecZ2Z-NEXT:   cufftExecZ2Z(plan /*cufftHandle*/, in /*cufftDoubleComplex **/,
// cufftExecZ2Z-NEXT:                out /*cufftDoubleComplex **/, dir /*int*/);
// cufftExecZ2Z-NEXT: Is migrated to:
// cufftExecZ2Z-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftExecZ2Z-NEXT:   plan = dpct::fft::fft_engine::create();
// cufftExecZ2Z-NEXT:   plan->compute<sycl::double2, sycl::double2>(in, out, dir == 1 ? dpct::fft::fft_direction::backward : dpct::fft::fft_direction::forward);
// cufftExecZ2Z-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftExecD2Z | FileCheck %s -check-prefix=cufftExecD2Z
// cufftExecD2Z: CUDA API:
// cufftExecD2Z-NEXT:   cufftHandle plan;
// cufftExecD2Z-NEXT:   cufftCreate(&plan /*cufftHandle **/);
// cufftExecD2Z-NEXT:   cufftExecD2Z(plan /*cufftHandle*/, in /*cufftDoubleReal **/,
// cufftExecD2Z-NEXT:                out /*cufftDoubleComplex **/);
// cufftExecD2Z-NEXT: Is migrated to:
// cufftExecD2Z-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftExecD2Z-NEXT:   plan = dpct::fft::fft_engine::create();
// cufftExecD2Z-NEXT:   plan->compute<double, sycl::double2>(in, out, dpct::fft::fft_direction::forward);
// cufftExecD2Z-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftExecR2C | FileCheck %s -check-prefix=cufftExecR2C
// cufftExecR2C: CUDA API:
// cufftExecR2C-NEXT:   cufftHandle plan;
// cufftExecR2C-NEXT:   cufftCreate(&plan /*cufftHandle **/);
// cufftExecR2C-NEXT:   cufftExecR2C(plan /*cufftHandle*/, in /*cufftReal **/,
// cufftExecR2C-NEXT:                out /*cufftComplex **/);
// cufftExecR2C-NEXT: Is migrated to:
// cufftExecR2C-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftExecR2C-NEXT:   plan = dpct::fft::fft_engine::create();
// cufftExecR2C-NEXT:   plan->compute<float, sycl::float2>(in, out, dpct::fft::fft_direction::forward);
// cufftExecR2C-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftPlan2d | FileCheck %s -check-prefix=cufftPlan2d
// cufftPlan2d: CUDA API:
// cufftPlan2d-NEXT:   cufftHandle plan;
// cufftPlan2d-NEXT:   cufftPlan2d(&plan /*cufftHandle **/, nx /*int*/, ny /*int*/,
// cufftPlan2d-NEXT:               type /*cufftType*/);
// cufftPlan2d-NEXT: Is migrated to:
// cufftPlan2d-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftPlan2d-NEXT:   plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), nx, ny, type);
// cufftPlan2d-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftPlan1d | FileCheck %s -check-prefix=cufftPlan1d
// cufftPlan1d: CUDA API:
// cufftPlan1d-NEXT:   cufftHandle plan;
// cufftPlan1d-NEXT:   cufftPlan1d(&plan /*cufftHandle **/, nx /*int*/, type /*cufftType*/,
// cufftPlan1d-NEXT:               num_of_trans /*int*/);
// cufftPlan1d-NEXT: Is migrated to:
// cufftPlan1d-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftPlan1d-NEXT:   plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), nx, type, num_of_trans);
// cufftPlan1d-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftPlan3d | FileCheck %s -check-prefix=cufftPlan3d
// cufftPlan3d: CUDA API:
// cufftPlan3d-NEXT:   cufftHandle plan;
// cufftPlan3d-NEXT:   cufftPlan3d(&plan /*cufftHandle **/, nx /*int*/, ny /*int*/, nz /*int*/,
// cufftPlan3d-NEXT:               type /*cufftType*/);
// cufftPlan3d-NEXT: Is migrated to:
// cufftPlan3d-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftPlan3d-NEXT:   plan = dpct::fft::fft_engine::create(&dpct::get_default_queue(), nx, ny, nz, type);
// cufftPlan3d-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftExecC2C | FileCheck %s -check-prefix=cufftExecC2C
// cufftExecC2C: CUDA API:
// cufftExecC2C-NEXT:   cufftHandle plan;
// cufftExecC2C-NEXT:   cufftCreate(&plan /*cufftHandle **/);
// cufftExecC2C-NEXT:   cufftExecC2C(plan /*cufftHandle*/, in /*cufftComplex **/,
// cufftExecC2C-NEXT:                out /*cufftComplex **/, dir /*int*/);
// cufftExecC2C-NEXT: Is migrated to:
// cufftExecC2C-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftExecC2C-NEXT:   plan = dpct::fft::fft_engine::create();
// cufftExecC2C-NEXT:   plan->compute<sycl::float2, sycl::float2>(in, out, dir == 1 ? dpct::fft::fft_direction::backward : dpct::fft::fft_direction::forward);
// cufftExecC2C-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftExecC2R | FileCheck %s -check-prefix=cufftExecC2R
// cufftExecC2R: CUDA API:
// cufftExecC2R-NEXT:   cufftHandle plan;
// cufftExecC2R-NEXT:   cufftCreate(&plan /*cufftHandle **/);
// cufftExecC2R-NEXT:   cufftExecC2R(plan /*cufftHandle*/, in /*cufftComplex **/,
// cufftExecC2R-NEXT:                out /*cufftReal **/);
// cufftExecC2R-NEXT: Is migrated to:
// cufftExecC2R-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftExecC2R-NEXT:   plan = dpct::fft::fft_engine::create();
// cufftExecC2R-NEXT:   plan->compute<sycl::float2, float>(in, out, dpct::fft::fft_direction::backward);
// cufftExecC2R-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftExecZ2D | FileCheck %s -check-prefix=cufftExecZ2D
// cufftExecZ2D: CUDA API:
// cufftExecZ2D-NEXT:   cufftHandle plan;
// cufftExecZ2D-NEXT:   cufftCreate(&plan /*cufftHandle **/);
// cufftExecZ2D-NEXT:   cufftExecZ2D(plan /*cufftHandle*/, in /*cufftDoubleComplex **/,
// cufftExecZ2D-NEXT:                out /*cufftDoubleReal **/);
// cufftExecZ2D-NEXT: Is migrated to:
// cufftExecZ2D-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftExecZ2D-NEXT:   plan = dpct::fft::fft_engine::create();
// cufftExecZ2D-NEXT:   plan->compute<sycl::double2, double>(in, out, dpct::fft::fft_direction::backward);
// cufftExecZ2D-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftDestroy | FileCheck %s -check-prefix=cufftDestroy
// cufftDestroy: CUDA API:
// cufftDestroy-NEXT:   cufftDestroy(plan /*cufftHandle*/);
// cufftDestroy-NEXT: Is migrated to:
// cufftDestroy-NEXT:   dpct::fft::fft_engine::destroy(plan);
// cufftDestroy-EMPTY:

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftCreate | FileCheck %s -check-prefix=cufftCreate
// cufftCreate: CUDA API:
// cufftCreate-NEXT:   cufftCreate(plan /*cufftHandle **/);
// cufftCreate-NEXT: Is migrated to:
// cufftCreate-NEXT:   *plan = dpct::fft::fft_engine::create();
// cufftCreate-EMPTY:

