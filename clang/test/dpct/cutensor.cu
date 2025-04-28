// UNSUPPORTED: cuda-8.0, cuda-9.2, cuda-10.0, cuda-10.2, cuda-11.1, cuda-11.2, cuda-11.3, cuda-11.4, cuda-11.5, cuda-11.6, cuda-11.7, cuda-12.0, cuda-12.1, cuda-12.2, cuda-12.3, cuda-12.4, cuda-12.5, cuda-12.6
// UNSUPPORTED: v8.0, v9.2, v10.0, v10.2, v11.1, v11.2, v11.3, v11.4, v11.5, v11.6, v11.7, v12.0, v12.1, v12.2, v12.3, v12.4, v12.5, v12.6
// RUN: dpct --format-range=none --out-root=%T/cutensor %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/cutensor/cutensor.dp.cpp
#include <cstdio>
#include <cutensor.h>
#include <cutensorMg.h>

int main() {
  cudaStream_t stream;

  // Basic handle creation and destruction
  cutensorHandle_t handle;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorCreate is not supported.
  cutensorCreate(&handle);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorDestroy is not supported.
  cutensorDestroy(handle);

  // Tensor descriptor
  cutensorTensorDescriptor_t tensorDesc;
  uint32_t numModes = 0;
  int64_t *extent;
  int64_t *stride;
  cutensorDataType_t tensorDataType;
  uint32_t alignmentRequirement;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorCreateTensorDescriptor is not supported.
  cutensorCreateTensorDescriptor(handle, &tensorDesc, numModes, extent, stride, tensorDataType, alignmentRequirement);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorDestroyTensorDescriptor is not supported.
  cutensorDestroyTensorDescriptor(tensorDesc);

  // Get cuTENSOR versions and error strings
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorGetErrorString is not supported.
  const char *errStr = cutensorGetErrorString(CUTENSOR_STATUS_SUCCESS);
  printf("Error String: %s\n", errStr);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorGetVersion is not supported.
  int version = cutensorGetVersion();
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorGetCudartVersion is not supported.
  int cudartVer = cutensorGetCudartVersion();
  printf("cuTENSOR Version: %d, CUDA Runtime Version: %d\n", version, cudartVer);

  // Elementwise trinary operations
  cutensorOperationDescriptor_t opDesc;
  int32_t *modes;
  cutensorOperator_t op;
  cutensorComputeDescriptor_t descCompute;
  cutensorPlan_t plan;
  const void *alpha, *A, *beta, *B, *gamma, *C;
  void *D;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorCreateElementwiseTrinary is not supported.
  cutensorCreateElementwiseTrinary(handle, &opDesc, tensorDesc, modes, op, tensorDesc, modes, op, tensorDesc, modes, op, tensorDesc, modes, op, op, descCompute);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorElementwiseTrinaryExecute is not supported.
  cutensorElementwiseTrinaryExecute(handle, plan, alpha, A, beta, B, gamma, C, D, stream);

  // Elementwise binary operations
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorCreateElementwiseBinary is not supported.
  cutensorCreateElementwiseBinary(handle, &opDesc, tensorDesc, modes, op, tensorDesc, modes, op, tensorDesc, modes, op, descCompute);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorElementwiseBinaryExecute is not supported.
  cutensorElementwiseBinaryExecute(handle, plan, alpha, A, gamma, C, D, stream);

  // Permutation
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorCreatePermutation is not supported.
  cutensorCreatePermutation(handle, &opDesc, tensorDesc, modes, op, tensorDesc, modes, descCompute);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorPermute is not supported.
  cutensorPermute(handle, plan, alpha, A, D, stream);

  // Contraction
  void *workspace;
  uint64_t workspaceSize;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorCreateContraction is not supported.
  cutensorCreateContraction(handle, &opDesc, tensorDesc, modes, op, tensorDesc, modes, op, tensorDesc, modes, op, tensorDesc, modes, descCompute);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorContract is not supported.
  cutensorContract(handle, plan, alpha, A, B, beta, C, D, workspace, workspaceSize, stream);

  // Contraction Trinary
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorCreateContractionTrinary is not supported.
  cutensorCreateContractionTrinary(handle, &opDesc, tensorDesc, modes, op, tensorDesc, modes, op, tensorDesc, modes, op, tensorDesc, modes, op, tensorDesc, modes, descCompute);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorContractTrinary is not supported.
  cutensorContractTrinary(handle, plan, alpha, A, B, C, beta, C, D, workspace, workspaceSize, stream);

  // Reduction
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorCreateReduction is not supported.
  cutensorCreateReduction(handle, &opDesc, tensorDesc, modes, op, tensorDesc, modes, op, tensorDesc, modes, op, descCompute);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorReduce is not supported.
  cutensorReduce(handle, plan, alpha, A, beta, C, D, workspace, workspaceSize, stream);

  // Operation descriptor functions
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorDestroyOperationDescriptor is not supported.
  cutensorDestroyOperationDescriptor(opDesc);
  cutensorOperationDescriptorAttribute_t opAttr;
  void *buf;
  size_t sizeInBytes;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorOperationDescriptorGetAttribute is not supported.
  cutensorOperationDescriptorGetAttribute(handle, opDesc, opAttr, buf, sizeInBytes);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorOperationDescriptorSetAttribute is not supported.
  cutensorOperationDescriptorSetAttribute(handle, opDesc, opAttr, buf, sizeInBytes);

  // Plan preference functions
  cutensorPlanPreference_t planPref;
  cutensorPlanPreferenceAttribute_t planPrefAttr;
  cutensorAlgo_t algo;
  cutensorJitMode_t jitMode;
  cutensorWorksizePreference_t workspacePref;
  uint64_t *workspaceSizeEstimate;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorCreatePlanPreference is not supported.
  cutensorCreatePlanPreference(handle, &planPref, algo, jitMode);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorDestroyPlanPreference is not supported.
  cutensorDestroyPlanPreference(planPref);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorPlanPreferenceSetAttribute is not supported.
  cutensorPlanPreferenceSetAttribute(handle, planPref, planPrefAttr, buf, sizeInBytes);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorEstimateWorkspaceSize is not supported.
  cutensorEstimateWorkspaceSize(handle, opDesc, planPref, workspacePref, workspaceSizeEstimate);
  cutensorPlanAttribute_t planAttr;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorCreatePlan is not supported.
  cutensorCreatePlan(handle, &plan, opDesc, planPref, *workspaceSizeEstimate);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorDestroyPlan is not supported.
  cutensorDestroyPlan(plan);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorPlanGetAttribute is not supported.
  cutensorPlanGetAttribute(handle, plan, planAttr, buf, sizeInBytes);

  // Plan cache management
  uint32_t numCachelinesRead;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorHandleResizePlanCache is not supported.
  cutensorHandleResizePlanCache(handle, 0);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorHandleReadPlanCacheFromFile is not supported.
  cutensorHandleReadPlanCacheFromFile(handle, "plan_cache.dat", &numCachelinesRead);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorHandleWritePlanCacheToFile is not supported.
  cutensorHandleWritePlanCacheToFile(handle, "plan_cache.dat");
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorReadKernelCacheFromFile is not supported.
  cutensorReadKernelCacheFromFile(handle, "kernel_cache.dat");
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorWriteKernelCacheToFile is not supported.
  cutensorWriteKernelCacheToFile(handle, "kernel_cache.dat");

  // Logger functions
  cutensorLoggerCallback_t callback;
  FILE *file;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorLoggerSetCallback is not supported.
  cutensorLoggerSetCallback(callback);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorLoggerSetFile is not supported.
  cutensorLoggerSetFile(file);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorLoggerOpenFile is not supported.
  cutensorLoggerOpenFile("logfile.txt");
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorLoggerSetLevel is not supported.
  cutensorLoggerSetLevel(0);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorLoggerSetMask is not supported.
  cutensorLoggerSetMask(0);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorLoggerForceDisable is not supported.
  cutensorLoggerForceDisable();

  // Multi-grid (Mg) APIs
  // Mg handle creation and destruction
  cutensorMgHandle_t mgHandle;
  uint32_t numDevices;
  const int32_t *devices;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgCreate is not supported.
  cutensorMgCreate(&mgHandle, numDevices, devices);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgDestroy is not supported.
  cutensorMgDestroy(mgHandle);

  // Mg tensor descriptor
  cutensorMgTensorDescriptor_t *desc;
  const int64_t *blockSize;
  const int32_t *deviceCount;
  cudaDataType_t dataType;
  cutensorMgTensorDescriptor_t mgTensorDesc;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgCreateTensorDescriptor is not supported.
  cutensorMgCreateTensorDescriptor(mgHandle, &mgTensorDesc, numModes, extent, stride, blockSize, stride, deviceCount, numDevices, devices, dataType);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgDestroyTensorDescriptor is not supported.
  cutensorMgDestroyTensorDescriptor(mgTensorDesc);

  // Mg copy descriptor & plan
  cutensorMgCopyDescriptor_t mgCopyDesc;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgCreateCopyDescriptor is not supported.
  cutensorMgCreateCopyDescriptor(mgHandle, &mgCopyDesc, mgTensorDesc, modes, mgTensorDesc, modes);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgDestroyCopyDescriptor is not supported.
  cutensorMgDestroyCopyDescriptor(mgCopyDesc);
  int64_t hostWorkspaceSize = 0;
  int64_t *deviceWorkspaceSize;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgCopyGetWorkspace is not supported.
  cutensorMgCopyGetWorkspace(mgHandle, mgCopyDesc, deviceWorkspaceSize, &hostWorkspaceSize);
  cutensorMgCopyPlan_t mgCopyPlan;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgCreateCopyPlan is not supported.
  cutensorMgCreateCopyPlan(mgHandle, &mgCopyPlan, mgCopyDesc, deviceWorkspaceSize, hostWorkspaceSize);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgDestroyCopyPlan is not supported.
  cutensorMgDestroyCopyPlan(mgCopyPlan);
  void *ptrDst;
  const void *ptrSrc;
  void *deviceWorkspace;
  void *hostWorkspace;
  cudaStream_t *streams;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgCopy is not supported.
  cutensorMgCopy(mgHandle, mgCopyPlan, &ptrDst, &ptrSrc, &deviceWorkspace, hostWorkspace, streams);

  // Mg contraction descriptor, find, plan, and execution
  cutensorMgContractionDescriptor_t mgContrDesc;
  cutensorComputeType_t computeType;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgCreateContractionDescriptor is not supported.
  cutensorMgCreateContractionDescriptor(mgHandle, &mgContrDesc, mgTensorDesc, modes, mgTensorDesc, modes, mgTensorDesc, modes, mgTensorDesc, modes, computeType);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgDestroyContractionDescriptor is not supported.
  cutensorMgDestroyContractionDescriptor(mgContrDesc);
  cutensorMgContractionFind_t mgContrFind;
  cutensorMgAlgo_t mgAlgo;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgCreateContractionFind is not supported.
  cutensorMgCreateContractionFind(mgHandle, &mgContrFind, mgAlgo);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgDestroyContractionFind is not supported.
  cutensorMgDestroyContractionFind(mgContrFind);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgContractionGetWorkspace is not supported.
  cutensorMgContractionGetWorkspace(mgHandle, mgContrDesc, mgContrFind, workspacePref, deviceWorkspaceSize, &hostWorkspaceSize);
  cutensorMgContractionPlan_t mgContrPlan;
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgCreateContractionPlan is not supported.
  cutensorMgCreateContractionPlan(mgHandle, &mgContrPlan, mgContrDesc, mgContrFind, deviceWorkspaceSize, hostWorkspaceSize);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgDestroyContractionPlan is not supported.
  cutensorMgDestroyContractionPlan(mgContrPlan);
  // CHECK: DPCT1007:{{[0-9]+}}: Migration of cutensorMgContraction is not supported.
  cutensorMgContraction(mgHandle, mgContrPlan, alpha, &A, &B, beta, &C, &D, &deviceWorkspace, hostWorkspace, streams);

  return 0;
}
