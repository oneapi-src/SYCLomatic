// UNSUPPORTED: system-windows
// RUN: rm -rf %T/build %T/source && mkdir -p %T/build/objs
// RUN: mkdir -p %T/build/objs
// RUN: mkdir %T/source
// RUN: cat %s > %T/source/foo.cu
// RUN: cat %S/bar.cpp > %T/source/bar.cpp
// RUN: cd %T/build
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"nvcc -c -std=c++20 -I %T -o %T/build/objs/foo.cu.o %T/source/foo.cu\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"/%T\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%T/source/foo.cu\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"g++ -c -std=c++11 -o %T/build/objs/bar.cpp.dp.o %T/source/bar.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"/%T\"," >> compile_commands.json
// RUN: echo "        \"file\": \"/%T/source/bar.cpp\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"ld objs/foo.cu.o objs/bar.cpp.dp.o -o app\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%T/build\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: cd %T
// RUN: dpct -in-root ./ -out-root out -p build/ -gen-build-script --cuda-include-path="%cuda-path/include"
// RUN: cat %S/Makefile.dpct.ref  >%T/Makefile.dpct.check
// RUN: cat %T/out/Makefile.dpct >> %T/Makefile.dpct.check
// RUN: FileCheck --match-full-lines --input-file %T/Makefile.dpct.check %T/Makefile.dpct.check

__global__ void foo() {
}

int main() {
	return 0;
}
