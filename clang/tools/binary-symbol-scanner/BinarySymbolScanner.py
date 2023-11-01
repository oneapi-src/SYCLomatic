#--------------------------------------------------------------------------------
# MIT License
# 
# Copyright (c) Intel Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#--------------------------------------------------------------------------------

# Description:
# Scanner to scrub binaries for symbols and filter them by library APIs
#
# Version 1.2.2
#
## Usage:- BinarySymbolScanner.py [-h] -c "company-name" -p "product-name" [-b "path"] [-d] [-s] [-x]
#
# The captured functions from the binaries in the search path will be saved in
# 'product-name.funcs.csv'. If statistics flag is turned on, function occurrence
# histogram will be saved to 'product-name.stats.csv'.
# 
# 
#  | Options/flags                                | Description                                              |
#  | -------------------------------------------- | ---------------------------------------------------------|
#  | -h, --help                                   | Show this help message and exit                          |
#  | -c COMPANY_NAME, --company-name COMPANY_NAME | Company name the product belongs to.                     |
#  | -p PROD_NAME, --product-name PROD_NAME       | Product name for which the binaries are being scanned.   |
#  | -b BIN_DIR, --binary-dir BIN_DIR             | Application binary directory to be scanned. If the       |
#  |                                              | option is not provided, the script will scan the         |
#  |                                              | current directory and all child directories for binaries.| 
#  | -d, --debug                                  | Enable debugging help.                                   |
#  | -s, --statistics                             | Create function histograms.                              |
#  | -x, --extend-non-intel                       | Extend the scanning to non-Intel performance libraries.  | 
#

import argparse
import os
import re
import sys
import subprocess
from pathlib import Path

# Commands to use to determine the symbols in the binaries on Windows and Linux
if os.name == "nt":
    os_name = "Windows"
    command = ["dumpbin", "/imports"]
else:
    if os.name == "posix":
        os_name = "Linux"
        command = ["nm", "--demangle"]
    else:
        sys.exit("Running script on an unsupported OS: Aborting! [Supported OSes - Windows, Linux]")


# Regular expression dictionary for all the function signatures from Intel oneAPI copmponents
regex_dict = {
    "IPP IPL": ["ipp[ir].*$", "ipp[A-Z].*$"],
    "IPP SPL": ["ipps.*$"],
    "IPP Computer Vision": ["ipp[cv|cc|ch|e|dc|cp].*$"],
    # MKL C interfaces
    "MKL C BLAS/BLAS-like extensions": ["cblas_."], # Include sparse BLAS routines
    "MKL Compact/Sparse/Support": ["mkl_."],
    "MKL Extended Eigensolver": ["[sdcz]feast_.", "feastinit"],
    "MKL VM Math": ["v[sdcz][A-Z].", "vm[sdcz].", "vml[A-Z].", "MKL."],
    "MKL Statistics": ["vsli[A-Z].", "vsl[A-Z].", "vsl[sdcz][A-Z].", "vRng[A-Z].", "v[sdi][A-D]"],
    "MKL FFT": ["Dfti[A-Z].", "dfti."],
    "MKL PBLAS": ["p[scdzi]."],
    "MKL PDE": ["[ds]_.", "free_trig_transform", "free_Hemholtz_.", "free_sph_." ],
    "MKL NL Opt. Solvers": ["[ds]trnlsp_.", "[ds]trnlspbc_.", "[ds]jacobi."],
    "MKL Support": ["xerbla", "pxerbla", "lsame.", "second", "dsecnd"],
    "MKL BLACS": ["blacs_.", "[isdcz]gam.", "[isdcz]gsum.", "[isdcz]gesd2d", "[isdcz]trsd2d", "[isdcz]gerv2d", 
                  "[isdcz]trrv2d", "[isdcz]gebs2d", "[isdcz]trbs2d", "[isdcz]gebr2d", "[isdcz]trbr2d"],
    "MKL Data Fitting": ["df[ds][A-Z].", "dfi[ds][A-Z].", "df[A-Z]."],
    "MKL C LAPACK": ["LAPACKE_.", "LAPACKE_[sdcz].", "[sdcz]g[gebt].", "[sdcz]dt.", "[sdcz]p[osfpbt][tcres].", 
                   "[sdcz]sy[egtcrs][qrofvw].", "[sdcz]h[efgpbs].", "[sdcz]s[fupbt].", "[sdcz]t[gzfrpb][tcrqmes][qvnxeyrofjt].", 
                   "[sdcz]or[bcgm].", "[sdcz]un[bcgm].", "[sdcz]b[db].", "[sdcz][ou]p[gm].", "[sdcz]disna.",
                  "[sdcz]la[cgknprstu].", "i[sdcz]m.", "ila[vem]."],
    "MKL ScaLAPACK": ["p[sdcz]g[egb].", "p[sdcz]d[bt].", "p[sdcz]p[obt].", "p[sdcz]t[rz].", "p[sdcz]or[gm].", 
                      "p[sdcz]un[gm].", "p[sdcz]sy[egnt].", "p[sdcz]he.", "p[sdcz]su.", "p[sdcz]la[bcehimnpqrstuw].", 
                      "p[sdcz]max.", "p[ijm].", "[sdcz]comb.", "[sdcz]la[hmpqrs].", "[sdcz][sdpt][btr][e].", 
                      "p[sdcz]r[os].", "descinit", "numroc"],
    "MKL PARDISO/Cluster Sparse/DSS/RCI ISS/LU/": ["pardiso.", "mkl_pardiso.", "cluster_sparse.", "dss_.", 
                                                   "dcg_.", "dcgmr[eh]s.", "dcsrilu.", "sparse_matrix_checker."],
    "MKL Graph": ["^.*mkl_graph_."],
    "MKL C++ BLAS": ["^.*oneapi::mkl::blas::."],
    "MKL C++ LAPACK": ["^.*oneapi::mkl::lapack::."],
    "MKL C++ RNG": ["^.*oneapi::mkl::rng."],
    "MKL C++ Summary/Stats": ["^.*oneapi::mkl::stats."],
    "MKL C++ FFT": ["^.*oneapi::mkl::dft::.", "^.*descriptor<."],
    "MKL C++ Data Fitting": ["^.*oneapi::mkl::experimental."],
    "MKL C++ Sparse BLAS": ["^.*oneapi::mkl::sparse::."],
    "MKL C++ Vector Math": ["^.*oneapi::mkl::vm::."],
    # Old MKL regex for FORTRAN
    "MKL C++ BLAS Level 1": ["asum", "axpy", "copy_$", "dot.", "iam.", "nrm2", "rot.", "scal", "sdsdot", "swap_$"],
    "MKL C++ BLAS Level 2": ["g[be]mv", "ger.", "h[be]mv", "her.", "hp.", "s[bpy]mv", "s[yp]r", "s[yp]r2", "t[bpr]mv", "t[bpr]sv"],
    "MKL C++ BLAS Level 3": ["[gh]emm", "her.", "symm", "syr2k", "syrk", "tr[sm]m"],
    "MKL C++ BLAS Extensions": ["axp.", "._batch", "gem.", "[oi]mat.", "syrk", "tr[sm]m"],
    "MKL Fortran VMM Arithmetic": ["add$", "sub$", "sqr$", "mul.", "conj$", "abs$", "arg$", "linear.", "fmod$", "remainder$"],
    "MKL Fortran Power and Root": ["inv$", "div$", "sqrt$", "inv.", "cbrt$", "pow.", "hypot"],
    "MKL Fortran VMM Exp and Log": ["exp.", "ln", "log."],
    "MKL Fortran VMM Trig./Hyperb.": ["cos.", "sin.", "tan.", "cis", "acos.", "asin.", "atan.", "cospi", "sinpi", "tanpi"],
    "MKL C++ VMM Special": ["erf.", "cdf.", "lgamma", "tgamma"],
    "MKL Fortran VMM Rounding": ["floor", "ceil", "trunc", "round", "nearbyint", "rint", "modf", "frac"],
    "MKL C++ VMM Service": ["set_mode", "get_mode", "set_status", "get_status", "clear_status", "create_error_handler"],
    "MKL C++ VMM Miscellaneous": ["copysign", "nextafter", "fdim", "fmax", "fmin", "maxmag", "minmag"],
    # MKL C ScaLAPACK/PBLAS onwards and FORTRAN not done
    "oneAPI DPL": ["^.*oneapi::dpl::."],
    "oneDNN": ["^.*dnnl::.", "^.*dnnl_."],
    "oneDAL": ["^.*oneapi::dal::."],
    "RT Embree": ["^.*rtc."],
    "RT Open VKL": ["^.*vkl."],
    "RT OSPRay": ["^.*osp."],
    "RT Open Image Denoise": ["^.*oidn."],
    "OpenCV Cuda": ["^.* cv::cuda::.", "^cv::cuda::."],
    "OpenCV": ["^.* cv::.", "^cv::.", "^.* hal_.", "^hal_."],
    "oneCCL": ["^.*oneapi::ccl::."],
    "LevelZero": ["ze[A-Z]."],
    "oneTBB": ["^.*oneapi::tbb::."],
    "Legacy TBB": ["tbb::."],
    "Intel OpenMP": ["kmp_.", "^.*kmpc_."],
    "SYCL": ["^.*sycl::."],
    "OpenCL": ["^.*cl[A-Z]."]
}

# Regular expression dictionary for all the function signatures of similar offerings
# from NVIDIA and AMD.
regex_ext_dict = {
    # NVIDIA Library signature regex
    "VPI": ["^.*vpi[A-Z]."],
    "cuFFT":  ["^.*cufft.", "^.*cufftdx::."],
    "cuBLAS": ["^.*cublas."],
    "cuSolver": ["^.*cusolver."],
    "cuSPARSE": ["^.*cusparse."],
    "cuRAND": ["^.*curand."],
    "nvJPEG": ["^.*nvjpeg."],
    "nvTIFF": ["^.*nvtiff."],
    "NPP":  ["^.*nppi.", "^.*npps"],
    "cuTENSOR": ["^.*cutensor."],
    "cuDNN": ["^.*cudnn."],
    "NCCL": ["^.*nccl."],
    "AmgX": ["^.*amgx::."],
    "Thrust": ["^.*thrust::."],
    "nvSHMEM": ["^.*nvshmem_."],
    "CUDA": ["^.*cuda.", "^.*make_cuda."],
    "CUB": ["^.*Cub[A-Z]."],
    "NVVM": ["^nvvm"],
    "TensorRT": ["^nvinfer1", "^nvcaffeparser1", "^nvonnxparser", "^nvuffparser"],
    # AMD library signature regex TO BE COMPLETED (hipeigen, etc)
    "rocBLAS": ["^.*rocblas_."],
    "rocRAND": ["^.*rocrand."],
    "rocFFT": ["^.*rocfft."],
    "rocSPARSE": ["^.*rocsparse."],
    "rocALUTION": ["^.*rocalution::."],
    "rocPRIM": ["^.*rocprim::."],
    "rocSMI": ["^.*rsmi_."],
    "HIP": ["^.*hip[A-Z].", "^.*make_hip.", "^.*hipblas.", "^.*hipsparse."],
    "MIOPEN": ["^.*miopen[A-Z]."]
}

def scan_functions_linux(binary: str, debug_output):
    functions = {}
    cmd = command + [binary]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in result.stdout.decode().splitlines():
        if debug_output:
            out_text = "========== All Functions:  '{}'"
            print(out_text.format(line))
        if not re.search(".@GLIB|.so", line):
            tokens = line.strip().split()
            token_count = len(tokens)
            if token_count > 0:
                if re.match("U|u|w|W", tokens[0]):
                    # Sometimes functions signatures can have spaces in them
                    # and will result in multiple tokens. We piece them back
                    # here 
                    func_sig = ""
                    for i in range(token_count):
                        if i > 0:
                            func_sig += tokens[i] + " "
                    if debug_output:
                       out_text = "---------- Recorded Function:  '{}'"
                       print(out_text.format(func_sig))
                    functions[func_sig] = False
                elif token_count > 1 and re.match("r|R|t|T|D|d", tokens[1]):
                    # We have defined symbols that may match with header only implementations
                    # or statically linked in functions
                    func_sig = ""
                    for i in range(token_count):
                        if i > 1:
                            func_sig += tokens[i] + " "
                    if re.search("oneapi::|cusp::|tbb::|sycl::|ipp|mkl|ze|cv::|dnnl::", func_sig):
                       if debug_output:
                          out_text = "---------- Special Function:  '{}'"
                          print(out_text.format(func_sig))
                       functions[func_sig] = False

    return functions

def demangle_function_name(func: str):
    # Check to see if the name is mangled - if so, it will 
    # begin with '?'
    if func.find('?') > -1:
        cmd = ["undname", "0x1002", func]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Windows output is multiple lines with Microsoft headers
        # and other unwanted text. Scan through these to get to the
        # demangled name
        for line in result.stdout.decode().splitlines():
            res_tokens = line.strip().split()
            token_count = len(res_tokens)
            if token_count > 0:
                if res_tokens[0] == 'is':
                   func = res_tokens[token_count-1]
                   # The string we get from 'undname' will be enclosed in quotes,
                   # so we need to remove the qotes before storing them
                   func = func.replace("\"", "")
                   return func
    #If we cannot demangle it, then return the mangled name
    return func

def scan_functions_windows(binary: str, debug_output):
    functions = {}
    cmd = command + [binary]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in result.stdout.decode().splitlines():
        tokens = line.strip().split()
        no_of_tks = len(tokens)
        if no_of_tks > 0:
           func = tokens[no_of_tks-1]
           # If the function name is mangled, we have to undecorate it 
           # on Windows. The utility on Linux demangles the name, so 
           # no additional work is needed
           if os.name == "nt" and not re.search(".dll|.exe", func):
               func = demangle_function_name(func)
           if debug_output:
              out_text = "******  '{}' <== [{}]"
              print(out_text.format(tokens[no_of_tks-1], func))
           functions[func] = False
    return functions

def scan_functions(binary, debug_output):
    if os.name == "nt":
        return scan_functions_windows(binary, debug_output)

    if os.name == "posix":
        return scan_functions_linux(binary, debug_output)
    return []

def is_hidden(dir, dir_lut):
    for key in dir_lut:
        if key in dir:
            return True
    return False

def scan_directory(path, file_dict):
    # Scan directories in provided path
    # 1. Record the hidden directories in the provided path
    # 2. Capture all files in current 'root' directory that don't begin 
    #    with '.' and all child directories that don't begin with '.'
    #    TBD: Not sure if this restriction should be placed, especially
    #    if applications use hidden directories to place files
    # 3. file_dict can now be used by binary scanners
    hidden_dir_lut = {}
    for root,d_names,f_names in os.walk(path):
        # Record the hidden directories
        for d in d_names:
            if d.startswith('.'):
                full_path = os.path.join(root, d)
                hidden_dir_lut[full_path] = True
        # Find the list of files in the current directory that are 
        # hidden files and put them in 'list'
        list = [];
        for f in f_names:
            if not f.startswith('.'):
               list.append(f)

        # If the current directory doesn't happen to be a hidden directory
        # or a child of a hidden directory, then update the dictionary with
        # the list of files scanned
        if not is_hidden(root, hidden_dir_lut):
            file_dict[root] = list

# Scan the regex patterns, compare to the symbols and save
# the matched symbols into a new dictionary that can be used
# to create the full lookup
def pattern_match_store(regex_dict, symbols, out_dict, debug):
    for lib_key in regex_dict:
        lib_funcs = []
        for pattern in regex_dict[lib_key]:
            for sym in symbols:
                if debug:
                    out_text = "---------- Symbol  '{}' <==  Pattern [{}] = {}"
                    print(out_text.format(sym, pattern, re.match(pattern, sym)))
                # This is now a state machine. If a symbol has matched one of the 
                # patterns, we don't try to match that symbol again
                if not symbols[sym] and re.match(pattern, sym):
                   if debug:
                      out_text = "---------- MATCH  '{}' <==  Pattern [{}] = {}"
                      print(out_text.format(sym, pattern, re.match(pattern, sym)))
                   lib_funcs.append(sym)
                   symbols[sym] = True
        out_dict[lib_key] = lib_funcs

def scan_files(dict, debug_output, extend_scan):
    # 1. 'dict' is the dictionary prepared by scan_directory()
    # 2. For each file in the list, scan it for symbols
    # 3. If the symbol matches any of the regular expressions, record it
    # 4. Filter some of the matches [This is just a workaround for now]
    # 5. Save the successful symbols in search_lut which will be used
    #    to print the information or generate a CSV file.
    search_lut = {}
    for key in dict:
        file_lut = {}
        for file in dict[key]:
            lib_lut = {}
            full_path = os.path.join(key, file)
            symbols = scan_functions(full_path, debug_output)
            # Test for Intel library signatures
            pattern_match_store(regex_dict, symbols, lib_lut, debug_output)
            if extend_scan:
                # Test for NVidia and AMD library signatures
                pattern_match_store(regex_ext_dict, symbols, lib_lut, debug_output)

            file_lut[file] = lib_lut
        search_lut[key] = file_lut
    return search_lut
    

def do_scrub(args):
    if not args.product_name:
        print("Product name must be provided: see help (-h)")
        return 1
    func_stats = {}
    debug_output = args.debug
    csv_fname = args.product_name.strip().replace(" ", "_")
    stats_fname = csv_fname + ".stats.csv"
    csv_fname += ".funcs.csv"
    non_intel = args.extend_non_intel
    if debug_output:
        print("Debugging output enabled for pattern matching!")
    file_dict = {}
    # Scan directories in binary directory
    scan_directory(args.binary_dir, file_dict)
    search_lut = scan_files(file_dict, debug_output, non_intel)

    path = Path(csv_fname)
    if path.is_file():
        print("Overwriting file: ", csv_fname)
    func_csv_f = open(csv_fname, "w")
    func_csv_f.write("sep=|\n")
    func_csv_f.write("Company|Product Name|OS|Binary|Component|Function Name\n")
    for dir in search_lut:
        files = search_lut[dir]
        for file in files:
            libs = files[file]
            for lib_key in libs:
                funcs = libs[lib_key]
                for func in funcs:
                    func_csv_f.write("%s|%s|%s|%s|%s|%s\n" %(args.company_name, args.product_name, os_name, file, lib_key, func))
                    if func.endswith('_'):
                        # FORTRAN function signatures have this, so remove the 
                        # trailing '_' for accurate statistics
                        mod_func = func[:-1]
                    else:
                        mod_func = func
                    if mod_func in func_stats:
                        func_stats[mod_func] += 1
                    else:
                        func_stats[mod_func] = 1
                    if debug_output:
                        out_text = "Function: {} <==  Count: {}"
                        print(out_text.format(mod_func, func_stats[mod_func]))
    func_csv_f.close()

    # Compile statistics for functions
    if args.statistics:
        path = Path(stats_fname)
        if path.is_file():
            print("Overwriting file: ", stats_fname)
        stats_csv_f = open(stats_fname, "w")
        stats_csv_f.write("sep=|\n")
        stats_csv_f.write("Function|Count\n")
        for func in func_stats:
            stats_csv_f.write("%s|%d\n" %(func, func_stats[func]))
        stats_csv_f.close()

    return 0


def main():
    curr_dir = os.getcwd()

    parser = argparse.ArgumentParser(prog="BinarySymbolScanner.py", 
                                     description=" The captured functions will be saved in 'PROD_NAME.funcs.csv'\n and if statistics flag is turned on, function use histograms will\n be saved to 'PROD_NAME.stats.csv'. ", 
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-c", "--company-name", required=True, metavar="COMPANY_NAME", help="Company name the product belongs to.")
    parser.add_argument("-p", "--product-name", required=True, metavar="PROD_NAME", help="Product name for which the binaries are being scanned.")
    parser.add_argument("-b", "--binary-dir", metavar="BIN_DIR", type=Path, default=curr_dir, help="Application binary directory to be scanned. If the\noption is not provided, the script will scan the current\ndirectory and all child directories for binaries.")
    parser.add_argument("-d", "--debug", action='store_true', help="Enable debugging help.")
    parser.add_argument("-s", "--statistics", action='store_true', help="Create function histograms.")
    parser.add_argument("-x", "--extend-non-intel", action='store_true', help="Extend the scanning to non-Intel performance libraries.")

    args = parser.parse_args()

    print("")
    print(" +--------------------------------------------------------------------+")
    if args.extend_non_intel:
        print(" | ** The script is currently enabled for scanning Intel/non-Intel    |")
        print(" |    performance library signatures.                                 |")
    else:
        print(" | ** The script is currently enabled for scanning Intel performance  |")
        print(" |    library signatures.                                             |")
    print(" +--------------------------------------------------------------------+")
    print("========================================================================")
    print("")


    return do_scrub(args)

if __name__ == "__main__":
    print("\n================== BinarySymbolScanner =================================")
    print("")
    print("  BinarySymbolScanner.py is a python script that can be used to scan")
    print("  the binaries in the specified directory (-b, --binary-dir) for")
    print("  performance library signatures from Intel. The script can also be")
    print("  used to scan non-Intel performance library signatures through the")
    print("  opt-in flag (-x, --extend-scan).")
    print("")
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)
