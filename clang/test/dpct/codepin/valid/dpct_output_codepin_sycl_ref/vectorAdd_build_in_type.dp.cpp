// RUN: echo 0
//CHECK: dpct::experimental::get_ptr_size_map()[d_A] = VECTOR_SIZE * sizeof(float);
//CHECK: dpct::experimental::get_ptr_size_map()[d_B] = VECTOR_SIZE * sizeof(float);
//CHECK: dpct::experimental::get_ptr_size_map()[d_C] = VECTOR_SIZE * sizeof(float);
//CHECK: dpct::experimental::gen_prolog_API_CP("{{[._0-9a-zA-Z\/\(\)\:]+}}", &q_ct1, "d_A", d_A, "d_B", d_B, "d_C", d_C);
//CHECK: dpct::experimental::gen_epilog_API_CP("{{[._0-9a-zA-Z\/\(\)\:]+}}", &q_ct1, "d_A", d_A, "d_B", d_B, "d_C", d_C);