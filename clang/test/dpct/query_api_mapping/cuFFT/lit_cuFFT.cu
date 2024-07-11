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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftPlan2d | FileCheck %s -check-prefix=cufftPlan2d
// cufftPlan2d: CUDA API:
// cufftPlan2d-NEXT:   cufftHandle plan;
// cufftPlan2d-NEXT:   cufftPlan2d(&plan /*cufftHandle **/, nx /*int*/, ny /*int*/,
// cufftPlan2d-NEXT:               type /*cufftType*/);
// cufftPlan2d-NEXT: Is migrated to:
// cufftPlan2d-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftPlan2d-NEXT:   plan = dpct::fft::fft_engine::create(&dpct::get_in_order_queue(), nx, ny, type);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftPlan1d | FileCheck %s -check-prefix=cufftPlan1d
// cufftPlan1d: CUDA API:
// cufftPlan1d-NEXT:   cufftHandle plan;
// cufftPlan1d-NEXT:   cufftPlan1d(&plan /*cufftHandle **/, nx /*int*/, type /*cufftType*/,
// cufftPlan1d-NEXT:               num_of_trans /*int*/);
// cufftPlan1d-NEXT: Is migrated to:
// cufftPlan1d-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftPlan1d-NEXT:   plan = dpct::fft::fft_engine::create(&dpct::get_in_order_queue(), nx, type, num_of_trans);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftPlan3d | FileCheck %s -check-prefix=cufftPlan3d
// cufftPlan3d: CUDA API:
// cufftPlan3d-NEXT:   cufftHandle plan;
// cufftPlan3d-NEXT:   cufftPlan3d(&plan /*cufftHandle **/, nx /*int*/, ny /*int*/, nz /*int*/,
// cufftPlan3d-NEXT:               type /*cufftType*/);
// cufftPlan3d-NEXT: Is migrated to:
// cufftPlan3d-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftPlan3d-NEXT:   plan = dpct::fft::fft_engine::create(&dpct::get_in_order_queue(), nx, ny, nz, type);

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

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftDestroy | FileCheck %s -check-prefix=cufftDestroy
// cufftDestroy: CUDA API:
// cufftDestroy-NEXT:   cufftDestroy(plan /*cufftHandle*/);
// cufftDestroy-NEXT: Is migrated to:
// cufftDestroy-NEXT:   dpct::fft::fft_engine::destroy(plan);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftCreate | FileCheck %s -check-prefix=cufftCreate
// cufftCreate: CUDA API:
// cufftCreate-NEXT:   cufftCreate(plan /*cufftHandle **/);
// cufftCreate-NEXT: Is migrated to:
// cufftCreate-NEXT:   *plan = dpct::fft::fft_engine::create();

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftEstimate1d | FileCheck %s -check-prefix=cufftEstimate1d
// cufftEstimate1d: CUDA API:
// cufftEstimate1d-NEXT:   cufftEstimate1d(nx /*int*/, type /*cufftType*/, num_of_trans /*int*/,
// cufftEstimate1d-NEXT:                   worksize /*size_t **/);
// cufftEstimate1d-NEXT: Is migrated to:
// cufftEstimate1d-NEXT:   dpct::fft::fft_engine::estimate_size(nx, type, num_of_trans, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftEstimate2d | FileCheck %s -check-prefix=cufftEstimate2d
// cufftEstimate2d: CUDA API:
// cufftEstimate2d-NEXT:   cufftEstimate2d(nx /*int*/, ny /*int*/, type /*cufftType*/,
// cufftEstimate2d-NEXT:                   worksize /*size_t **/);
// cufftEstimate2d-NEXT: Is migrated to:
// cufftEstimate2d-NEXT:   dpct::fft::fft_engine::estimate_size(nx, ny, type, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftEstimate3d | FileCheck %s -check-prefix=cufftEstimate3d
// cufftEstimate3d: CUDA API:
// cufftEstimate3d-NEXT:   cufftEstimate3d(nx /*int*/, ny /*int*/, nz /*int*/, type /*cufftType*/,
// cufftEstimate3d-NEXT:                   worksize /*size_t **/);
// cufftEstimate3d-NEXT: Is migrated to:
// cufftEstimate3d-NEXT:   dpct::fft::fft_engine::estimate_size(nx, ny, nz, type, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftEstimateMany | FileCheck %s -check-prefix=cufftEstimateMany
// cufftEstimateMany: CUDA API:
// cufftEstimateMany-NEXT:   cufftEstimateMany(dim /*int*/, n /*int **/, inembed /*int **/,
// cufftEstimateMany-NEXT:                     istride /*int*/, idist /*int*/, onembed /*int **/,
// cufftEstimateMany-NEXT:                     ostride /*int*/, odist /*int*/, type /*cufftType*/,
// cufftEstimateMany-NEXT:                     num_of_trans /*int*/, worksize /*size_t **/);
// cufftEstimateMany-NEXT: Is migrated to:
// cufftEstimateMany-NEXT:   dpct::fft::fft_engine::estimate_size(dim, n, inembed, istride, idist, onembed, ostride, odist, type, num_of_trans, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftGetProperty | FileCheck %s -check-prefix=cufftGetProperty
// cufftGetProperty: CUDA API:
// cufftGetProperty-NEXT:   cufftGetProperty(type /*libraryPropertyType*/, value /*int **/);
// cufftGetProperty-NEXT: Is migrated to:
// cufftGetProperty-NEXT:   dpct::mkl_get_version(type, value);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftGetSize | FileCheck %s -check-prefix=cufftGetSize
// cufftGetSize: CUDA API:
// cufftGetSize-NEXT:   cufftGetSize(plan /*cufftHandle*/, worksize /*size_t **/);
// cufftGetSize-NEXT: Is migrated to:
// cufftGetSize-NEXT:   plan->get_workspace_size(worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftGetSize1d | FileCheck %s -check-prefix=cufftGetSize1d
// cufftGetSize1d: CUDA API:
// cufftGetSize1d-NEXT:   cufftGetSize1d(plan /*cufftHandle*/, nx /*int*/, type /*cufftType*/,
// cufftGetSize1d-NEXT:                  num_of_trans /*int*/, worksize /*size_t **/);
// cufftGetSize1d-NEXT: Is migrated to:
// cufftGetSize1d-NEXT:   dpct::fft::fft_engine::estimate_size(nx, type, num_of_trans, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftGetSize2d | FileCheck %s -check-prefix=cufftGetSize2d
// cufftGetSize2d: CUDA API:
// cufftGetSize2d-NEXT:   cufftGetSize2d(plan /*cufftHandle*/, nx /*int*/, ny /*int*/,
// cufftGetSize2d-NEXT:                  type /*cufftType*/, worksize /*size_t **/);
// cufftGetSize2d-NEXT: Is migrated to:
// cufftGetSize2d-NEXT:   dpct::fft::fft_engine::estimate_size(nx, ny, type, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftGetSize3d | FileCheck %s -check-prefix=cufftGetSize3d
// cufftGetSize3d: CUDA API:
// cufftGetSize3d-NEXT:   cufftGetSize3d(plan /*cufftHandle*/, nx /*int*/, ny /*int*/, nz /*int*/,
// cufftGetSize3d-NEXT:                  type /*cufftType*/, worksize /*size_t **/);
// cufftGetSize3d-NEXT: Is migrated to:
// cufftGetSize3d-NEXT:   dpct::fft::fft_engine::estimate_size(nx, ny, nz, type, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftGetSizeMany | FileCheck %s -check-prefix=cufftGetSizeMany
// cufftGetSizeMany: CUDA API:
// cufftGetSizeMany-NEXT:   cufftGetSizeMany(plan /*cufftHandle*/, dim /*int*/, n /*int **/,
// cufftGetSizeMany-NEXT:                    inembed /*int **/, istride /*int*/, idist /*int*/,
// cufftGetSizeMany-NEXT:                    onembed /*int **/, ostride /*int*/, odist /*int*/,
// cufftGetSizeMany-NEXT:                    type /*cufftType*/, num_of_trans /*int*/,
// cufftGetSizeMany-NEXT:                    worksize /*size_t **/);
// cufftGetSizeMany-NEXT: Is migrated to:
// cufftGetSizeMany-NEXT:   dpct::fft::fft_engine::estimate_size(dim, n, inembed, istride, idist, onembed, ostride, odist, type, num_of_trans, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftGetSizeMany64 | FileCheck %s -check-prefix=cufftGetSizeMany64
// cufftGetSizeMany64: CUDA API:
// cufftGetSizeMany64-NEXT:   cufftGetSizeMany64(plan /*cufftHandle*/, dim /*int*/, n /*long long int **/,
// cufftGetSizeMany64-NEXT:                      inembed /*long long int **/, istride /*long long int*/,
// cufftGetSizeMany64-NEXT:                      idist /*long long int*/, onembed /*long long int **/,
// cufftGetSizeMany64-NEXT:                      ostride /*long long int*/, odist /*long long int*/,
// cufftGetSizeMany64-NEXT:                      type /*cufftType*/, num_of_trans /*long long int*/,
// cufftGetSizeMany64-NEXT:                      worksize /*size_t **/);
// cufftGetSizeMany64-NEXT: Is migrated to:
// cufftGetSizeMany64-NEXT:   dpct::fft::fft_engine::estimate_size(dim, n, inembed, istride, idist, onembed, ostride, odist, type, num_of_trans, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftGetVersion | FileCheck %s -check-prefix=cufftGetVersion
// cufftGetVersion: CUDA API:
// cufftGetVersion-NEXT:   cufftGetVersion(version /*int **/);
// cufftGetVersion-NEXT: Is migrated to:
// cufftGetVersion-NEXT:   dpct::mkl_get_version(dpct::version_field::major, version);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftMakePlan1d | FileCheck %s -check-prefix=cufftMakePlan1d
// cufftMakePlan1d: CUDA API:
// cufftMakePlan1d-NEXT:   cufftMakePlan1d(plan /*cufftHandle*/, nx /*int*/, type /*cufftType*/,
// cufftMakePlan1d-NEXT:                   num_of_trans /*int*/, worksize /*size_t **/);
// cufftMakePlan1d-NEXT: Is migrated to:
// cufftMakePlan1d-NEXT:   plan->commit(&dpct::get_in_order_queue(), nx, type, num_of_trans, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftMakePlan2d | FileCheck %s -check-prefix=cufftMakePlan2d
// cufftMakePlan2d: CUDA API:
// cufftMakePlan2d-NEXT:   cufftMakePlan2d(plan /*cufftHandle*/, nx /*int*/, ny /*int*/,
// cufftMakePlan2d-NEXT:                   type /*cufftType*/, worksize /*size_t **/);
// cufftMakePlan2d-NEXT: Is migrated to:
// cufftMakePlan2d-NEXT:   plan->commit(&dpct::get_in_order_queue(), nx, ny, type, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftMakePlan3d | FileCheck %s -check-prefix=cufftMakePlan3d
// cufftMakePlan3d: CUDA API:
// cufftMakePlan3d-NEXT:   cufftMakePlan3d(plan /*cufftHandle*/, nx /*int*/, ny /*int*/, nz /*int*/,
// cufftMakePlan3d-NEXT:                   type /*cufftType*/, worksize /*size_t **/);
// cufftMakePlan3d-NEXT: Is migrated to:
// cufftMakePlan3d-NEXT:   plan->commit(&dpct::get_in_order_queue(), nx, ny, nz, type, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftMakePlanMany | FileCheck %s -check-prefix=cufftMakePlanMany
// cufftMakePlanMany: CUDA API:
// cufftMakePlanMany-NEXT:   cufftMakePlanMany(plan /*cufftHandle*/, dim /*int*/, n /*int **/,
// cufftMakePlanMany-NEXT:                     inembed /*int **/, istride /*int*/, idist /*int*/,
// cufftMakePlanMany-NEXT:                     onembed /*int **/, ostride /*int*/, odist /*int*/,
// cufftMakePlanMany-NEXT:                     type /*cufftType*/, num_of_trans /*int*/,
// cufftMakePlanMany-NEXT:                     worksize /*size_t **/);
// cufftMakePlanMany-NEXT: Is migrated to:
// cufftMakePlanMany-NEXT:   plan->commit(&dpct::get_in_order_queue(), dim, n, inembed, istride, idist, onembed, ostride, odist, type, num_of_trans, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftMakePlanMany64 | FileCheck %s -check-prefix=cufftMakePlanMany64
// cufftMakePlanMany64: CUDA API:
// cufftMakePlanMany64-NEXT:   cufftMakePlanMany64(plan /*cufftHandle*/, dim /*int*/, n /*long long int **/,
// cufftMakePlanMany64-NEXT:                       inembed /*long long int **/, istride /*long long int*/,
// cufftMakePlanMany64-NEXT:                       idist /*long long int*/, onembed /*long long int **/,
// cufftMakePlanMany64-NEXT:                       ostride /*long long int*/, odist /*long long int*/,
// cufftMakePlanMany64-NEXT:                       type /*cufftType*/, num_of_trans /*long long int*/,
// cufftMakePlanMany64-NEXT:                       worksize /*size_t **/);
// cufftMakePlanMany64-NEXT: Is migrated to:
// cufftMakePlanMany64-NEXT:   plan->commit(&dpct::get_in_order_queue(), dim, n, inembed, istride, idist, onembed, ostride, odist, type, num_of_trans, worksize);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftPlanMany | FileCheck %s -check-prefix=cufftPlanMany
// cufftPlanMany: CUDA API:
// cufftPlanMany-NEXT:   cufftHandle plan;
// cufftPlanMany-NEXT:   cufftPlanMany(&plan /*cufftHandle **/, dim /*int*/, n /*int **/,
// cufftPlanMany-NEXT:                 inembed /*int **/, istride /*int*/, idist /*int*/,
// cufftPlanMany-NEXT:                 onembed /*int **/, ostride /*int*/, odist /*int*/,
// cufftPlanMany-NEXT:                 type /*cufftType*/, num_of_trans /*int*/);
// cufftPlanMany-NEXT: Is migrated to:
// cufftPlanMany-NEXT:   dpct::fft::fft_engine_ptr plan;
// cufftPlanMany-NEXT:   plan = dpct::fft::fft_engine::create(&dpct::get_in_order_queue(), dim, n, inembed, istride, idist, onembed, ostride, odist, type, num_of_trans);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftSetAutoAllocation | FileCheck %s -check-prefix=cufftSetAutoAllocation
// cufftSetAutoAllocation: CUDA API:
// cufftSetAutoAllocation-NEXT:   cufftSetAutoAllocation(plan /*cufftHandle*/, autoallocate /*int*/);
// cufftSetAutoAllocation-NEXT: Is migrated to:
// cufftSetAutoAllocation-NEXT:   plan->use_internal_workspace(autoallocate);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftSetStream | FileCheck %s -check-prefix=cufftSetStream
// cufftSetStream: CUDA API:
// cufftSetStream-NEXT:   cufftSetStream(plan /*cufftHandle*/, s /*cudaStream_t*/);
// cufftSetStream-NEXT: Is migrated to:
// cufftSetStream-NEXT:   plan->set_queue(s);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftSetWorkArea | FileCheck %s -check-prefix=cufftSetWorkArea
// cufftSetWorkArea: CUDA API:
// cufftSetWorkArea-NEXT:   cufftSetWorkArea(plan /*cufftHandle*/, workspace /*void **/);
// cufftSetWorkArea-NEXT: Is migrated to:
// cufftSetWorkArea-NEXT:   plan->set_workspace(workspace);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftXtExec | FileCheck %s -check-prefix=cufftXtExec
// cufftXtExec: CUDA API:
// cufftXtExec-NEXT:   cufftXtExec(plan /*cufftHandle*/, in /*void **/, out /*void **/, dir /*int*/);
// cufftXtExec-NEXT: Is migrated to:
// cufftXtExec-NEXT:   plan->compute<void, void>(in, out, dir == 1 ? dpct::fft::fft_direction::backward : dpct::fft::fft_direction::forward);

// RUN: dpct --cuda-include-path="%cuda-path/include" --query-api-mapping=cufftXtMakePlanMany | FileCheck %s -check-prefix=cufftXtMakePlanMany
// cufftXtMakePlanMany: CUDA API:
// cufftXtMakePlanMany-NEXT:   cufftXtMakePlanMany(plan /*cufftHandle*/, dim /*int*/, n /*long long int **/,
// cufftXtMakePlanMany-NEXT:                       inembed /*long long int **/, istride /*long long int*/,
// cufftXtMakePlanMany-NEXT:                       idist /*long long int*/, itype /*cudaDataType*/,
// cufftXtMakePlanMany-NEXT:                       onembed /*long long int **/, ostride /*long long int*/,
// cufftXtMakePlanMany-NEXT:                       odist /*long long int*/, otype /*cudaDataType*/,
// cufftXtMakePlanMany-NEXT:                       num_of_trans /*long long int*/, worksize /*size_t **/,
// cufftXtMakePlanMany-NEXT:                       exectype /*cudaDataType*/);
// cufftXtMakePlanMany-NEXT: Is migrated to:
// cufftXtMakePlanMany-NEXT:   plan->commit(&dpct::get_in_order_queue(), dim, n, inembed, istride, idist, itype, onembed, ostride, odist, otype, num_of_trans, worksize);
