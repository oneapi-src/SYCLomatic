// Option: --use-experimental-features=bindless_images
void test() {
  CUmipmappedArray mmArray;
  CUarray level_arr;
  // Start
  cuMipmappedArrayGetLevel(&level_arr/*CUarray **/, mmArray/*CUmipmappedArray*/, 1/*unsigned int*/);
  // End
}