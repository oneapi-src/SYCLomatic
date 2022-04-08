// RUN: c2s -out-root 0 %s %s --cuda-include-path="%cuda-path/include"

// Checking if C2S will crash when migrating this file twice in the same command line
template<class T>
struct SharedMem{
    __device__ void foo(){
        __shared__ int a;
    }
};

