// RUN: echo 0
//CHECK: dpctexp::codepin::get_ptr_size_map()[d_A] = VECTOR_SIZE * sizeof(float);
//CHECK: dpctexp::codepin::get_ptr_size_map()[d_B] = VECTOR_SIZE * sizeof(float);
//CHECK: dpctexp::codepin::get_ptr_size_map()[d_C] = VECTOR_SIZE * sizeof(float);
//CHECK: dpctexp::codepin::gen_prolog_API_CP("{{[._0-9a-zA-Z\/\(\)\:\-]+}}", 0, "d_A", d_A, "d_B", d_B, "d_C", d_C);
//CHECK: dpctexp::codepin::gen_epilog_API_CP("{{[._0-9a-zA-Z\/\(\)\:\-]+}}", 0, "d_A", d_A, "d_B", d_B, "d_C", d_C);
