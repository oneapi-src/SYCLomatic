// UNSUPPORTED: system-windows
// RUN: dpct --format-range=none --out-root %T/share_mem %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/share_mem/share_mem.dp.cpp --match-full-lines %s

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#define DATA_SIZE 1024

#define shName "shared_memory"

typedef struct sharedMemoryInfo_st {
  void *addr;
  size_t size;
  int shmFd;
} sharedMemoryInfo;

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info) {
  int status = 0;
  info->size = sz;
  info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  status = ftruncate(info->shmFd, sz);
  if (status != 0) {
    return status;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return 0;
}

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info) {
  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return 0;
}

typedef struct shmStruct_st {
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaIpcMemHandle_t is not supported, please try to remigrate with option: --use-experimental-features=level_zero.
  // CHECK-NEXT: */
  cudaIpcMemHandle_t memHandle;
} shmStruct;

__global__ void simpleKernel(int *ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[idx] = idx - 10;
}

typedef pid_t Process;
int spawnProcess(Process *process, const char *app, char *const *args) {
  *process = fork();
  if (*process == 0) {
    if (0 > execvp(app, args)) {
      return errno;
    }
  } else if (*process < 0) {
    return errno;
  }
  return 0;
}

int childProcess(int id) {
  int threads = 256;
  sharedMemoryInfo info;
  shmStruct *shm = NULL;
  if (sharedMemoryCreate(shName, sizeof(shmStruct), &info) != 0) {
    printf("Failed to create shared memory slab\n");
    exit(EXIT_FAILURE);
  }
  shm = (shmStruct *)info.addr;
  int *ptr;
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaIpcOpenMemHandle is not supported, please try to remigrate with option: --use-experimental-features=level_zero.
  // CHECK-NEXT: */
  cudaIpcOpenMemHandle((void **)&ptr, shm->memHandle,
                       cudaIpcMemLazyEnablePeerAccess);
  // CHECK: /*
  // CHECK-NEXT: DPCT1049:{{[0-9]+}}: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
  // CHECK-NEXT: */
  simpleKernel<<<1, threads, 0>>>(ptr);
  return 0;
}

int parentProcess(char *app) {

  shmStruct *shm;
  sharedMemoryInfo info;
  void *ptr;
  if (sharedMemoryCreate(shName, sizeof(*shm), &info) != 0) {
    printf("Failed to create shared memory slab\n");
    exit(EXIT_FAILURE);
  }
  shm = (shmStruct *)info.addr;
  memset((void *)shm, 0, sizeof(*shm));
  cudaMalloc(&ptr, DATA_SIZE);
  int *hostptr = (int *)malloc(DATA_SIZE);
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaIpcGetMemHandle is not supported, please try to remigrate with option: --use-experimental-features=level_zero.
  // CHECK-NEXT: */
  cudaIpcGetMemHandle(&shm->memHandle, ptr);

  char *const args[] = {app, "0", NULL};
  Process process;
  spawnProcess(&process, app, args);
  wait(NULL);

  cudaMemcpy(hostptr, ptr, DATA_SIZE, cudaMemcpyDeviceToHost);
  for (int i = 0; i < DATA_SIZE / sizeof(int); i++) {
    if (hostptr[i] != i - 10) {
      std::cout << "Error: " << hostptr[i] << " != " << i - 10 << "\n";
      return -1;
    }
  }
  std::cout << "verified Pass.\n";
  // CHECK: /*
  // CHECK-NEXT: DPCT1119:{{[0-9]+}}: Migration of cudaIpcCloseMemHandle is not supported, please try to remigrate with option: --use-experimental-features=level_zero.
  // CHECK-NEXT: */
  cudaIpcCloseMemHandle(ptr);
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 1) {
    return parentProcess(argv[0]);
  } else {
    return childProcess(atoi(argv[1]));
  }
}
