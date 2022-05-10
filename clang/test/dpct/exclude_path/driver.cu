// "Input Folder Structure:
// test/
// └── lib_1
// │   └── inc
// │   │   └── header_1.cuh // included by libsrc_1.cu, mian.cu
// │   └── src
// │       └── libsrc_1.cu
// └── lib_2
// │   └── inc
// │   │   └── header_2.cuh // included by libsrc_2.cu
// │   └── src
// │       └── libsrc_2.cu
// └── main.cu
// └── README.md
// └── compile_commands.json

// Compilation database:
// nvcc lib_1/src/libsrc_1.cu
// nvcc main.cu"

// RUN: echo "[" > %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc ./lib_1/src/libsrc_1.cu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%S/test_src\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"./lib_1/src/libsrc_1.cu\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc ./main.cu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%S/test_src\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"./main.cu\"" >> %T/compile_commands.json
// RUN: echo "    }" >> %T/compile_commands.json
// RUN: echo "]" >> %T/compile_commands.json

// RUN: sed -i 's/\\/\//g' %T/compile_commands.json

// RUN: dpct -p %T -in-root %S/test_src -out-root %T/test1 --in-root-exclude=%S/test_src/lib_1/inc --cuda-include-path="%cuda-path/include"
// test/
// └── lib_1
// │   └── src
// │       └── libsrc_1.dp.cpp
// └── main.dp.cpp
// └── MainSourceFiles.yaml
// RUN: bash %S/check_script.sh %T/test1/lib_1/inc/header_1.cuh %T
// RUN: bash %S/check_script.sh %T/test1/lib_1/inc/header_1.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test1/lib_1/src/libsrc_1.cu %T
// RUN: bash %S/check_script.sh %T/test1/lib_1/src/libsrc_1.dp.cpp %T
// RUN: bash %S/check_script.sh %T/test1/lib_2/inc/header_2.cuh %T
// RUN: bash %S/check_script.sh %T/test1/lib_2/inc/header_2.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test1/lib_2/src/libsrc_2.cu %T
// RUN: bash %S/check_script.sh %T/test1/lib_2/src/libsrc_2.dp.cpp %T

// RUN: dpct -p %T -in-root %S/test_src -out-root %T/test2 --in-root-exclude=%S/test_src/lib_1/src --cuda-include-path="%cuda-path/include"
// test/
// └── lib_1
// │   └── inc
// │       └── header_1.dp.hpp
// │       └── header_1.dp.hpp.yaml
// └── main.dp.cpp
// └── MainSourceFiles.yaml
// RUN: bash %S/check_script.sh %T/test2/lib_1/inc/header_1.cuh %T
// RUN: bash %S/check_script.sh %T/test2/lib_1/inc/header_1.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test2/lib_1/src/libsrc_1.cu %T
// RUN: bash %S/check_script.sh %T/test2/lib_1/src/libsrc_1.dp.cpp %T
// RUN: bash %S/check_script.sh %T/test2/lib_2/inc/header_2.cuh %T
// RUN: bash %S/check_script.sh %T/test2/lib_2/inc/header_2.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test2/lib_2/src/libsrc_2.cu %T
// RUN: bash %S/check_script.sh %T/test2/lib_2/src/libsrc_2.dp.cpp %T

// RUN: dpct -p %T -in-root %S/test_src -out-root %T/test3 --in-root-exclude=%S/test_src/lib_1/inc --process-all --cuda-include-path="%cuda-path/include"
// test/
// └── lib_1
// │   └── src
// │       └── libsrc_1.dp.cpp
// └── lib_2
// │   └── inc
// │   │   └── header_2.dp.hpp
// │   │   └── header_2.dp.hpp.yaml
// │   └── src
// │       └── libsrc_2.dp.cpp
// └── main.dp.cpp
// └── README.md
// └── compile_commands.json
// └── MainSourceFiles.yaml
// RUN: bash %S/check_script.sh %T/test3/lib_1/inc/header_1.cuh %T
// RUN: bash %S/check_script.sh %T/test3/lib_1/inc/header_1.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test3/lib_1/src/libsrc_1.cu %T
// RUN: bash %S/check_script.sh %T/test3/lib_1/src/libsrc_1.dp.cpp %T
// RUN: bash %S/check_script.sh %T/test3/lib_2/inc/header_2.cuh %T
// RUN: bash %S/check_script.sh %T/test3/lib_2/inc/header_2.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test3/lib_2/src/libsrc_2.cu %T
// RUN: bash %S/check_script.sh %T/test3/lib_2/src/libsrc_2.dp.cpp %T

// RUN: dpct -p %T -in-root %S/test_src -out-root %T/test4 --in-root-exclude=%S/test_src/lib_1/src --process-all --cuda-include-path="%cuda-path/include"
// test/
// └── lib_1
// │   └── inc
// │       └── header_1.dp.hpp
// │       └── header_1.dp.hpp.yaml
// └── lib_2
// │   └── inc
// │   │   └── header_2.dp.hpp
// │   │   └── header_2.dp.hpp.yaml
// │   └── src
// │       └── libsrc_2.dp.cpp
// └── main.dp.cpp
// └── README.md
// └── compile_commands.json
// └── MainSourceFiles.yaml
// RUN: bash %S/check_script.sh %T/test4/lib_1/inc/header_1.cuh %T
// RUN: bash %S/check_script.sh %T/test4/lib_1/inc/header_1.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test4/lib_1/src/libsrc_1.cu %T
// RUN: bash %S/check_script.sh %T/test4/lib_1/src/libsrc_1.dp.cpp %T
// RUN: bash %S/check_script.sh %T/test4/lib_2/inc/header_2.cuh %T
// RUN: bash %S/check_script.sh %T/test4/lib_2/inc/header_2.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test4/lib_2/src/libsrc_2.cu %T
// RUN: bash %S/check_script.sh %T/test4/lib_2/src/libsrc_2.dp.cpp %T

// RUN: dpct -p %T -in-root %S/test_src -out-root %T/test5 --in-root-exclude=%S/test_src/lib_2/inc --process-all --cuda-include-path="%cuda-path/include"
// test/
// └── lib_1
// │   └── inc
// │   │   └── header_1.dp.hpp
// │   │   └── header_1.dp.hpp.yaml
// │   └── src
// │       └── libsrc_1.dp.cpp
// └── lib_2
// │   └── src
// │       └── libsrc_2.dp.cpp
// └── main.dp.cpp
// └── README.md
// └── compile_commands.json
// └── MainSourceFiles.yaml
// RUN: bash %S/check_script.sh %T/test5/lib_1/inc/header_1.cuh %T
// RUN: bash %S/check_script.sh %T/test5/lib_1/inc/header_1.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test5/lib_1/src/libsrc_1.cu %T
// RUN: bash %S/check_script.sh %T/test5/lib_1/src/libsrc_1.dp.cpp %T
// RUN: bash %S/check_script.sh %T/test5/lib_2/inc/header_2.cuh %T
// RUN: bash %S/check_script.sh %T/test5/lib_2/inc/header_2.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test5/lib_2/src/libsrc_2.cu %T
// RUN: bash %S/check_script.sh %T/test5/lib_2/src/libsrc_2.dp.cpp %T

// RUN: dpct -p %T -in-root %S/test_src -out-root %T/test6 --in-root-exclude=%S/test_src/lib_2/src --process-all --cuda-include-path="%cuda-path/include"
// test/
// └── lib_1
// │   └── inc
// │   │   └── header_1.dp.hpp
// │   │   └── header_1.dp.hpp.yaml
// │   └── src
// │       └── libsrc_1.dp.cpp
// └── lib_2
// │   └── inc
// │       └── header_2.cuh
// │
// └── main.dp.cpp
// └── README.md
// └── compile_commands.json
// └── MainSourceFiles.yaml
// RUN: bash %S/check_script.sh %T/test6/lib_1/inc/header_1.cuh %T
// RUN: bash %S/check_script.sh %T/test6/lib_1/inc/header_1.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test6/lib_1/src/libsrc_1.cu %T
// RUN: bash %S/check_script.sh %T/test6/lib_1/src/libsrc_1.dp.cpp %T
// RUN: bash %S/check_script.sh %T/test6/lib_2/inc/header_2.cuh %T
// RUN: bash %S/check_script.sh %T/test6/lib_2/inc/header_2.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test6/lib_2/src/libsrc_2.cu %T
// RUN: bash %S/check_script.sh %T/test6/lib_2/src/libsrc_2.dp.cpp %T

// RUN: dpct -p %T -in-root %S/test_src -out-root %T/test7 --in-root-exclude=%S/test_src/lib_1/src --in-root-exclude=%S/test_src/lib_2/src --process-all --cuda-include-path="%cuda-path/include"
// test/
// └── lib_1
// │   └── inc
// │       └── header_1.dp.hpp
// │       └── header_1.dp.hpp.yaml
// └── lib_2
// │   └── inc
// │       └── header_2.cuh
// └── main.dp.cpp
// └── README.md
// └── compile_commands.json
// └── MainSourceFiles.yaml
// RUN: bash %S/check_script.sh %T/test7/lib_1/inc/header_1.cuh %T
// RUN: bash %S/check_script.sh %T/test7/lib_1/inc/header_1.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test7/lib_1/src/libsrc_1.cu %T
// RUN: bash %S/check_script.sh %T/test7/lib_1/src/libsrc_1.dp.cpp %T
// RUN: bash %S/check_script.sh %T/test7/lib_2/inc/header_2.cuh %T
// RUN: bash %S/check_script.sh %T/test7/lib_2/inc/header_2.dp.hpp %T
// RUN: bash %S/check_script.sh %T/test7/lib_2/src/libsrc_2.cu %T
// RUN: bash %S/check_script.sh %T/test7/lib_2/src/libsrc_2.dp.cpp %T

// RUN: dpct -p %T -in-root %S/test_src -out-root %T/test8 --in-root-exclude=%S/test_src/README.md --process-all --cuda-include-path="%cuda-path/include"
// test/
// └── lib_1
// │   └── inc
// │   │   └── header_1.dp.hpp
// │   │   └── header_1.dp.hpp.yaml
// │   └── src
// │       └── libsrc_1.dp.cpp
// └── lib_2
// │   └── inc
// │   │   └── header_2.dp.hpp
// │   │   └── header_2.dp.hpp.yaml
// │   └── src
// │       └── libsrc_2.dp.cpp
// └── main.dp.cpp
// └── compile_commands.json
// └── MainSourceFiles.yaml
// RUN: bash %S/check_script.sh %T/test8/README.md %T

// RUN: FileCheck --input-file %T/exist_check --match-full-lines %S/ref
