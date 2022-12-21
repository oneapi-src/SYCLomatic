// RUN: dpct --format-range=none -out-root %T/cufft-workspace %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only -std=c++14
// RUN: FileCheck --input-file %T/cufft-workspace/cufft-workspace.dp.cpp --match-full-lines %s
#include <cstdio>
#include <cufft.h>
#include <cuda_runtime.h>


int main() {
  cufftHandle plan;
  int rank;
  long long int *n_ll;
  long long int *inembed_ll;
  long long int istride_ll;
  long long int idist_ll;
  long long int *onembed_ll;
  long long int ostride_ll;
  long long int odist_ll;
  cufftType type;
  long long int batch_ll;
  size_t *workSize;

  int *n;
  int *inembed;
  int istride;
  int idist;
  int *onembed;
  int ostride;
  int odist;
  int batch;

  
  // CHECK: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "dpct::fft::fft_engine::estimate_size" is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::fft::fft_engine::estimate_size(n[0], type, batch, workSize);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "dpct::fft::fft_engine::estimate_size" is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::fft::fft_engine::estimate_size(n[0], n[1], type, workSize);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "dpct::fft::fft_engine::estimate_size" is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::fft::fft_engine::estimate_size(n[0], n[1], n[2], type, workSize);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "dpct::fft::fft_engine::estimate_size" is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::fft::fft_engine::estimate_size(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
  cufftEstimate1d(n[0], type, batch, workSize);
  cufftEstimate2d(n[0], n[1], type, workSize);
  cufftEstimate3d(n[0], n[1], n[2], type, workSize);
  cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);

  cufftCreate(&plan);
  // CHECK: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "dpct::fft::fft_engine::estimate_size" is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::fft::fft_engine::estimate_size(n[0], type, batch, workSize);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "dpct::fft::fft_engine::estimate_size" is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::fft::fft_engine::estimate_size(n[0], n[1], type, workSize);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "dpct::fft::fft_engine::estimate_size" is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::fft::fft_engine::estimate_size(n[0], n[1], n[2], type, workSize);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "dpct::fft::fft_engine::estimate_size" is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::fft::fft_engine::estimate_size(rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "dpct::fft::fft_engine::estimate_size" is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: dpct::fft::fft_engine::estimate_size(rank, n_ll, inembed_ll, istride_ll, idist_ll, onembed_ll, ostride_ll, odist_ll, type, batch_ll, workSize);
  cufftGetSize1d(plan, n[0], type, batch, workSize);
  cufftGetSize2d(plan, n[0], n[1], type, workSize);
  cufftGetSize3d(plan, n[0], n[1], n[2], type, workSize);
  cufftGetSizeMany(plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch, workSize);
  cufftGetSizeMany64(plan, rank, n_ll, inembed_ll, istride_ll, idist_ll, onembed_ll, ostride_ll, odist_ll, type, batch_ll, workSize);

  // CHECK: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: plan->use_internal_workspace(0);
  cufftSetAutoAllocation(plan, 0);
  // CHECK: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: /*
  // CHECK-NEXT: DPCT1099:{{[0-9]+}}: Verify if the default value of the direction and placement used in the function "commit" is correct.
  // CHECK-NEXT: */
  // CHECK-NEXT: plan->commit(&dpct::get_default_queue(), n[0], type, batch, workSize);
  cufftMakePlan1d(plan, n[0], type, batch, workSize);
  // CHECK: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: plan->get_workspace_size(workSize);
  cufftGetSize(plan, workSize);

  void *workArea;
  // CHECK: /*
  // CHECK-NEXT: DPCT1100:{{[0-9]+}}: Currently the DFT external workspace feature in the Intel(R) oneAPI Math Kernel Library (oneMKL) is only supported for GPU devices. Use the internal workspace if your code should run on non-GPU devices.
  // CHECK-NEXT: */
  // CHECK-NEXT: plan->set_workspace(workArea);
  cufftSetWorkArea(plan, workArea);

  cufftDestroy(plan);
  return 0;
}
