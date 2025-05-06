// UNSUPPORTED: system-windows
// RUN: dpct --use-experimental-features=level_zero --format-range=none  -out-root %T/share_mem_exp_option %s --cuda-include-path="%cuda-path/include" --sycl-named-lambda -- -x cuda --cuda-host-only
// RUN: FileCheck --input-file %T/share_mem_exp_option/share_mem_exp_option.dp.cpp --match-full-lines %s
// RUN: %if build_lit %{icpx -c -fsycl  -DNO_BUILD_TEST  %T/share_mem_exp_option/share_mem_exp_option.dp.cpp -o %T/share_mem_exp_option/share_mem_exp_option.dp.o %}

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define DATA_SIZE 1024
constexpr int N = 4096;
constexpr int ITERATIONS = 10;
constexpr int BLOCK_SIZE = 16;
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
  // CHECK: dpct::experimental::ipc_mem_handle_ext_t memHandle;
  cudaIpcMemHandle_t memHandle;
  // CHECK: dpct::experimental::ipc_event_pool_handle_ext_t eventHandle;
  cudaIpcEventHandle_t eventHandle;
} shmStruct;

__global__ void longKernel(float *matrixA, float *matrixB, float *matrixC, int *ptr) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N && j < N) {
    float sum = 0.0f;

    for (int k = 0; k < N; ++k) {
      volatile float a = matrixA[i * N + k];
      volatile float b = matrixB[k * N + j];
      sum += a * b;
    }

    for (int iter = 0; iter < ITERATIONS; ++iter) {
      sum = sqrtf(sum) + sinf(sum);
    }

    matrixC[i * N + j] = sum;
  }
  ptr[j] = j - 10;
}

__global__ void simpleKernel(int *ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[idx] = idx - 10;

  float temp = 0.0f;
  for (int j = 0; j < 1000000; ++j) {
    temp += sin(static_cast<float>(j)) * cos(static_cast<float>(j));
  }
}

__global__ void simpleKernel_2(int *ptr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[idx] = ptr[idx] + 10;
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
  cudaEvent_t event;
  cudaStream_t stream;
  shmStruct *shm = NULL;

  float *d_matrixA, *d_matrixB, *d_matrixC;
  size_t size = N * N * sizeof(float);

  cudaMalloc(&d_matrixA, size);
  cudaMalloc(&d_matrixB, size);
  cudaMalloc(&d_matrixC, size);

  float *h_matrixA = new float[N * N]{1.0f};
  float *h_matrixB = new float[N * N]{2.0f};

  cudaMemcpy(d_matrixA, h_matrixA, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_matrixB, h_matrixB, size, cudaMemcpyHostToDevice);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  if (sharedMemoryCreate(shName, sizeof(shmStruct), &info) != 0) {
    printf("Failed to create shared memory slab\n");
    exit(EXIT_FAILURE);
  }
  shm = (shmStruct *)info.addr;
  int *ptr;
  // CHECK: dpct::experimental::open_mem_ipc_handle(*(dpct::experimental::ipc_mem_handle_ext_t *)&shm->memHandle, (void **)&ptr);
  cudaIpcOpenMemHandle((void **)&ptr, *(cudaIpcMemHandle_t *)&shm->memHandle,
                       cudaIpcMemLazyEnablePeerAccess);
  // CHECK: dpct::experimental::open_event_pool_ipc_handle(&event, *(dpct::experimental::ipc_event_pool_handle_ext_t *)&shm->eventHandle);
  cudaIpcOpenEventHandle(
      &event, *(cudaIpcEventHandle_t *)&shm->eventHandle);

  longKernel<<<grid, block>>>(d_matrixA, d_matrixB, d_matrixC, ptr);

  cudaEventRecord(event, stream);
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
  // CHECK: dpct::experimental::get_mem_ipc_handle(ptr, (dpct::experimental::ipc_mem_handle_ext_t *)&shm->memHandle);
  cudaIpcGetMemHandle((cudaIpcMemHandle_t *)&shm->memHandle, ptr);

  cudaEvent_t event;
  cudaEventCreate(
      &event, cudaEventDisableTiming | cudaEventInterprocess);
  // CHECK: dpct::experimental::get_event_pool_ipc_handle(event, (dpct::experimental::ipc_event_pool_handle_ext_t *)&shm->eventHandle);
  cudaIpcGetEventHandle(
      (cudaIpcEventHandle_t *)&shm->eventHandle, event);

  char *const args[] = {app, "0", NULL};
  Process process;
  spawnProcess(&process, app, args);
  wait(NULL);

  cudaMemcpyAsync(hostptr, ptr, DATA_SIZE, cudaMemcpyDeviceToHost);
  for (int i = 0; i < DATA_SIZE / sizeof(int); i++) {
    if (hostptr[i] != i - 10) {
      std::cout << "Error: " << hostptr[i] << " != " << i - 10 << "\n";
      return -1;
    }
  }
  std::cout << "verified Pass.\n";
  // CHECK: zeMemCloseIpcHandle(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dpct::get_current_device().get_context()), ptr);
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
