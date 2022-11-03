====================================================================================================================

**Current libcurl version: curl-7.86.0**

====================================================================================================================
* Steps to update libcurl
********************************************************************************************************************
1 Download the target version of the libcurl from  https://curl.se/download.html

2 Linux build
```
   $./configure --prefix=/path/to/linux-install --without-ssl --without-ca-fallback --without-libidn2 --without-nghttp2 --without-libidn2 --without-mbedtls --without-zlib --disable-ldap --disable-ldaps --without-libgsas

   $ make -j

   $ make install
```

3  Windows build
```
   a. start Visual Studio X64 command line
   b. cd curl-7.86.0\winbuild
   c. change /MD to /MT and /MDd to /MTd in file winbuild\MakefileBuild.vc in line404 and line405
   e. nmake /f Makefile.vc mode=static ENABLE_SSPI=no ENABLE_IPV6=no ENABLE_IDN=no  ENABLE_SCHANNEL=no
   f. check the result in curl-7.86.0\builds folder
```

4. Update header files
```
   a. Copy header files from curl-7.86.0\builds\libcurl-vc-x64-release-static\include to clang/lib/DPCT/libcurl/include folder
```

5. Update static library
```
   a. Copy libcurl.a to clang/lib/DPCT/libcurl/lib/linux
   b. Copy libcurl_a.lib to clang/lib/DPCT/libcurl/lib/win
 ```


**Note: Make sure the build folder path doesn't include any senstive info. Build the libraries in the root fodler, such as D:\lib_curl.**
