// RUN: dpct -out-root %T/miscellaneous_apis %s --cuda-include-path="%cuda-path/include"
// RUN: FileCheck --match-full-lines --input-file %T/miscellaneous_apis/miscellaneous_apis.dp.cpp %s
// RUN: %if build_lit %{icpx -c -fsycl %T/miscellaneous_apis/miscellaneous_apis.dp.cpp -o %T/miscellaneous_apis/miscellaneous_apis.dp.o %}

void foo(const void **table, CUuuid *tableId) {
  // CHECK: /*
  // CHECK-NEXT: DPCT1026:{{[0-9]+}}: The call to cuGetExportTable was removed because the data of the retrieved table is insufficent to populate the function pointer with the necessary API. Check and implement the functionality correspounding to the API.
  // CHECK-NEXT: */
  cuGetExportTable(table, tableId);
}
