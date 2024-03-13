// UNSUPPORTED: system-windows
// RUN: cd %T
// RUN: mkdir predefined_macro_check_db
// RUN: cd predefined_macro_check_db
// RUN: cat %s > predefined_macro_check_db.cu
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"nvcc predefined_macro_check_db.cu\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/predefined_macro_check_db\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/predefined_macro_check_db/predefined_macro_check_db.cu\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: dpct -p=. --out-root=./out --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/predefined_macro_check_db/out/predefined_macro_check_db.dp.cpp
// RUN: %if build_lit %{icpx -c -fsycl %T/predefined_macro_check_db/out/predefined_macro_check_db.dp.cpp -o %T/predefined_macro_check_db/out/predefined_macro_check_db.o %}
// RUN: cd ..
// RUN: rm -rf ./predefined_macro_check_db

// CHECK: #ifdef DPCT_COMPATIBILITY_TEMP
#ifdef __NVCC__
void fun() {}
#else
@error "error"
#endif
