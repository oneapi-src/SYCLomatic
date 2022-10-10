// RUN: dpct --format-range=none -out-root %T/nccl_cudnn_fakeapi_test %s --cuda-include-path="%cuda-path/include" -- -x cuda --cuda-host-only
int fake__cudnn_call(){
 return 1;
}
int fake__nccl_call(){
 return 1;
}

void test(){
fake__cudnn_call();
fake__nccl_call();
}

class _cudnn_t {
 int a;
 int b;
};
_cudnn_t ab1;

__global__ void hello(){}

