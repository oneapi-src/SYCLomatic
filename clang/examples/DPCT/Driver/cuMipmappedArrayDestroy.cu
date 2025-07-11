// Option: --use-experimental-features=bindless_images
void test() {
    CUmipmappedArray mmArray;
    // Start
    cuMipmappedArrayDestroy(mmArray/*CUmipmappedArray*/);
    // End
}