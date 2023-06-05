// UNSUPPORTED: system-linux
// RUN: cd %T
// RUN: mkdir language_note_db_win
// RUN: cd language_note_db_win
// RUN: cat %s > language_note_db_win.cpp
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"CudaCompile language_note_db_win.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/language_note_db_win\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/language_note_db_win/language_note_db_win.cpp\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: dpct -p=. --out-root=./out --cuda-include-path="%cuda-path/include" > log.txt
// RUN: FileCheck %s --match-full-lines --input-file %T/language_note_db_win/out/language_note_db_win.cpp.dp.cpp -check-prefix=CODE
// RUN: FileCheck %s --match-full-lines --input-file %T/language_note_db_win/log.txt -check-prefix=LOG
// RUN: cd ..
// RUN: rm -rf ./language_note_db_win

// CODE: sycl::float2 f2;
float2 f2;

// LOG: NOTE: {{(.+)}}/language_note_db_win.cpp is treated as a CUDA file by default. Use the --extra-arg=-xc++ option to treat {{(.+)}}/language_note_db_win.cpp as a C++ file if needed.
