// UNSUPPORTED: -windows-
// intercept-build only supports Linux
// RUN: cp %S/build_log.txt %T/
// RUN: cp %S/bar.cu %T/
// RUN: cp %S/foo.cpp %T/
// RUN: cd %T

// ----- Test to use option '--work-directory' and option '--parse-build-log'
// RUN: intercept-build -vvv  --parse-build-log ./build_log.txt   --work-directory=./
// RUN: cat %S/compile_commands.json_ref  >%T/check_compilation_db.txt
// RUN: cat %T/compile_commands.json >>%T/check_compilation_db.txt
// RUN: FileCheck --match-full-lines --input-file %T/check_compilation_db.txt %T/check_compilation_db.txt
// RUN: dpct --format-range=none -p=./ -out-root %T/out %T/bar.cu --cuda-include-path="%cuda-path/include"
// RUN: FileCheck  %T/bar.cu  --match-full-lines --input-file %T/out/bar.dp.cpp

// ----- Test to use default value of option '--work-directory'
// RUN: rm compile_commands.json
// RUN: rm -rf ./out
// RUN: intercept-build -vvv  --parse-build-log ./build_log.txt
// RUN: dpct --format-range=none -p=./ -out-root %T/out %T/bar.cu --cuda-include-path="%cuda-path/include"
// RUN: FileCheck  %T/bar.cu  --match-full-lines --input-file %T/out/bar.dp.cpp

// ----- Test option '--work-directory' is not set correctly
// RUN: rm compile_commands.json
// RUN: intercept-build -vvv  --parse-build-log ./build_log.txt --work-directory=./out > output.log || true
// RUN: grep "option --work-directory is not set correctly" ./output.log


#ifdef __BAR__  // This macro is used to check whether compilation database is used
//CHECK: void bar(){}
__global__ void bar(){}
#endif
