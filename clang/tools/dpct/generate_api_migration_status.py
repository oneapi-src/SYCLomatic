import argparse
import os
import sys
import warnings
import time
import string
import csv
import re

output_file_suffix = 'md'
keep_only_ask_once = False
keep_file = True

pattern_re = re.compile("DPCT\d{4}")

Partial_re = re.compile("Partial", re.I)

def diag_DPCT_link(DPCT_diag:str):
    return "https://www.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/diagnostics-reference/"+DPCT_diag.lower()+".html"

def format_diagnostic_info_md(DPCT_diag_number:list, is_supported:bool):
    if not DPCT_diag_number:
        return ""
    return " / ".join(("["+diag_number+"]("+diag_DPCT_link(diag_number)+")" for diag_number in DPCT_diag_number if is_supported and diag_number != "DPCT1007"))

def format_diagnostic_info_csv(DPCT_diag_number:list, is_supported:bool):
    if not DPCT_diag_number:
        return ""
    return " / ".join((":ref:`"+diag_number+"`" for diag_number in DPCT_diag_number if is_supported and diag_number != "DPCT1007"))

format_diagnostic_info = {"md":format_diagnostic_info_md, "csv":format_diagnostic_info_csv}

def get_user_answer(msg: str):
    msg += ' [y/n]'
    while(True):
        ans = str(input(msg))
        if ans == 'y' or ans == 'Y':
            return True
        elif ans == 'n' or ans == 'N':
            return False


def get_output_filename(path_str: str, filename: str, keep_filename: bool):
    if keep_filename is True:
        return os.path.join(path_str, filename+'.'+output_file_suffix)
    new_file_name = filename + '_' + \
        time.strftime('%Y-%m-%d %H:%M:%S')+output_file_suffix
    return get_output_filename(path_str, new_file_name, True)


def open_output_file(path_str: str, filename: str):
    global keep_only_ask_once
    global keep_file
    filename_with_path = os.path.join(path_str, filename+'.'+output_file_suffix)
    if os.path.exists(filename_with_path) is True:
        if keep_only_ask_once is False:
            keep_file = get_user_answer(
                'output files exist, whether it can be overridden?')
            keep_only_ask_once = True
        filename_with_path = get_output_filename(path_str, filename, keep_file)
    print("Output file is "+filename_with_path)
    fout = open(filename_with_path, 'w')
    return fout


def format_sing_line(API_list: list):
    return '|' + '{0: <60}'.format(API_list[0]) + '|' + '{0: ^10}'.format(API_list[1]) + '|' + '{0: ^150}'.format(API_list[2]) + '|\n'


def format_lib(lib_name: str, APIs_list: list):
    lib_str = '# '+lib_name+'\n'
    lib_str += '| Function | Migration support | Diagnostic message|\n'
    lib_str += '| :---- | :----: | :--- |\n'
    for API_list in APIs_list:
        lib_str += format_sing_line(API_list)
    lib_str += '\n'
    return lib_str


def parse_macro_entry(line: str):
    line_list = line.split()
    isMemberAPI = 0
    if(line_list[0]=="ENTRY"):
        API_list = [line_list[1]]
        isMemberAPI = 0
    else:
        API_list = [line_list[1]+'::'+line_list[3]]
        isMemberAPI = 2
    is_supported = False
    if(line_list[3 + isMemberAPI] == "true"):
        API_list.append("YES")
        is_supported = True
    elif (line_list[3 + isMemberAPI] == "false"):
        is_supported = False
        API_list.append("NO")
    else:
        warnings.warn(
            "internal error: can not tell whether API is supported or not.")
        API_list.append("UNKNOW")
    DPCT_dia_msg = []
    Partial_msg = []
    for i in line_list[4 + isMemberAPI::] :
        DPCT_dia_msg += pattern_re.findall(i)
        Partial_msg +=  Partial_re.findall(i)
    if("DPCT1030" in DPCT_dia_msg):
        API_list[-1] = "NO"
        is_supported = False
    if(DPCT_dia_msg or Partial_msg):
        API_list.append(format_diagnostic_info[output_file_suffix](DPCT_dia_msg, is_supported)+(" Partial" if Partial_msg else ""))
    else:
        API_list.append('')
    return API_list


def get_API_status_list(file_lib: str):
    APIs_list = []
    with open(file_lib, 'r') as flib:
        for line in flib.readlines():
            img_file = line.strip()
            if(img_file == "" or img_file[0:5] != "ENTRY"):
                continue
            img_file = img_file.translate(str.maketrans('"(),', '    '))
            API_list = parse_macro_entry(img_file)
            APIs_list.append(API_list)
    return APIs_list


def update_lib_md(lib: str, file_lib: str, output_path: str):
    output_filename = lib+'_API_migration_status'
    APIs_list = get_API_status_list(file_lib)
    lib_str = format_lib(lib, APIs_list)
    fout = open_output_file(output_path, output_filename)
    if(fout.writable()):
        fout.write(lib_str)
        fout.close()
        return True
    else:
        warnings.warn(
            "output file can not be generated, lost status information about "+lib)
        fout.close()
        return False


def update_lib_csv(lib: str, file_lib: str, output_path: str):
    output_filename = lib+'_API_migration_status'
    APIs_list = get_API_status_list(file_lib)
    fout = open_output_file(output_path, output_filename)
    heading = ['Function','Migration Support','Diagnostic Message']
    headwriter = csv.DictWriter(fout, fieldnames = heading)
    headwriter.writeheader()
    writer = csv.writer(fout)
    if(fout.writable()):
        writer.writerows(APIs_list)
        fout.close()
        return True
    else:
        warnings.warn(
            "output file can not be generated, lost status information about "+lib)
        fout.close()
        return False


def do_update(args):
    update_func = {'md': update_lib_md,
                   'csv': update_lib_csv}
    global output_file_suffix
    res = True
    output_path = args.output_path
    SYCLomatic_path = args.SYCLomatic_path
    output_file_suffix = args.output_file_format
    DPCT_lib_path = os.path.join(SYCLomatic_path, 'clang', 'lib', 'DPCT')
    if os.path.exists(output_path) is False:
        warnings.warn("output path is not exist")
        return False
    lib_names = ['Runtime_and_Driver', 'CUB', 'cuBLAS', 'cuDNN', 'cuFFT', 'nvGRAPH', 'ASM',
                 'cuRAND', 'cuSOLVER', 'cuSPARSE', 'NCCL', 'nvJPEG', 'NVML', 'thrust', 'wmma']
    # lib file name = APINames_$(libname).inc
    for lib_name in lib_names:
        if lib_name == 'Runtime_and_Driver':
            lib_record_file = 'APINames.inc'
        else:
            lib_record_file = 'APINames'+'_'+lib_name+'.inc'
        lib_file = os.path.join(DPCT_lib_path, lib_record_file)
        if os.path.exists(lib_file) is False:
            warnings.warn(lib_file+" is not exist")
            continue
        res = res & update_func[output_file_suffix](
            lib_name, lib_file, output_path)
    return res


def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description="A script to get the API migration support status in Compatibility Tools",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output-path", help="Set the path of the output file", default=os.getcwd())
    parser.add_argument("--SYCLomatic-path", help="Set the path of the SYCLomatic",
                        default=os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    parser.add_argument("--output-file-format",
                        help="Set the format of output files, only 'md' and 'csv' are available", default="md")

    args = parser.parse_args()
    return do_update(args)


if __name__ == "__main__":
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)