// UNSUPPORTED: -windows-
// RUN: mkdir -p %T/build/objs
// RUN: mkdir %T/source
// RUN: cat %s > %T/source/foo.cu
// RUN: cd %T/build
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"nvcc -c  -o /%T/build/objs/foo.cu.o /%T/source/foo.cu\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"/%T\"," >> compile_commands.json
// RUN: echo "        \"file\": \"/%T/source/foo.cu\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"ld objs/foo.cu.o -o foo\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"/%T/build\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: cd %T
// RUN: dpct -in-root ./ -out-root out -p build/ -gen-build-script
// RUN: cat %S/Makefile.dpct.ref  >%T/Makefile.dpct.check
// RUN: cat %T/out/Makefile.dpct >> %T/Makefile.dpct.check
// RUN: FileCheck --match-full-lines --input-file %T/Makefile.dpct.check %T/Makefile.dpct.check

__global__ void foo() {
}

int main() {
	return 0;
}
