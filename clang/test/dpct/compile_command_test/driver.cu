// RUN: echo "[" > %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/t.cu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/t.cu\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/t.cpp\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/t.cpp\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/t2.cu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/t2.cu\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/t2.cpp\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/t2.cpp\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/t3.cu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/t3.cu\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/t3.cpp\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/t3.cpp\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"nvcc %S/t4.cu\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/t4.cu\"" >> %T/compile_commands.json
// RUN: echo "    }," >> %T/compile_commands.json
// RUN: echo "    {" >> %T/compile_commands.json
// RUN: echo "        \"command\": \"c++ %S/t4.cpp\"," >> %T/compile_commands.json
// RUN: echo "        \"directory\": \"%T\"," >> %T/compile_commands.json
// RUN: echo "        \"file\": \"%S/t4.cpp\"" >> %T/compile_commands.json
// RUN: echo "    }" >> %T/compile_commands.json
// RUN: echo "]" >> %T/compile_commands.json

// RUN: sed -i  '3,5s/\\/\//g'  %T/compile_commands.json
// RUN: sed -i  '8,10s/\\/\//g'  %T/compile_commands.json
// RUN: sed -i  '13,15s/\\/\//g'  %T/compile_commands.json
// RUN: sed -i  '18,20s/\\/\//g'  %T/compile_commands.json
// RUN: sed -i  '23,25s/\\/\//g'  %T/compile_commands.json
// RUN: sed -i  '28,30s/\\/\//g'  %T/compile_commands.json
// RUN: sed -i  '33,35s/\\/\//g'  %T/compile_commands.json
// RUN: sed -i  '38,40s/\\/\//g'  %T/compile_commands.json

// RUN: c2s -process-all -in-root=%S -out-root=%T -p=%T --cuda-include-path="%cuda-path/include"

// RUN: FileCheck %S/t.cpp --match-full-lines --input-file %T/t.cpp
// RUN: FileCheck %S/t2.cpp --match-full-lines --input-file %T/t2.cpp
// RUN: FileCheck %S/t3.cpp --match-full-lines --input-file %T/t3.cpp
// RUN: FileCheck %S/t4.cpp --match-full-lines --input-file %T/t4.cpp
// RUN: FileCheck %S/t.cu --match-full-lines --input-file %T/t.dp.cpp
// RUN: FileCheck %S/t2.cu --match-full-lines --input-file %T/t2.dp.cpp
// RUN: FileCheck %S/t3.cu --match-full-lines --input-file %T/t3.dp.cpp
// RUN: FileCheck %S/t4.cu --match-full-lines --input-file %T/t4.dp.cpp
