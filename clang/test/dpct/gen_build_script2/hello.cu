// UNSUPPORTED: system-windows
// RUN: cat %s > %T/hello.cu
// RUN: cat %S/Makefile > %T/Makefile
// RUN: cd %T/
// RUN: intercept-build /usr/bin/make -B
// RUN: dpct -in-root ./ -out-root out -p ./ -gen-build-script --cuda-include-path="%cuda-path/include"
// RUN: cat %S/Makefile.dpct.ref  >%T/Makefile.dpct.check
// RUN: cat %T/out/Makefile.dpct >> %T/Makefile.dpct.check
// RUN: FileCheck --match-full-lines --input-file %T/Makefile.dpct.check %T/Makefile.dpct.check
// RUN: %if build_lit %{cd out && make -f Makefile.dpct %}

// CHECK: void hello() {}
__global__ void hello() {}

int main() {
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}
