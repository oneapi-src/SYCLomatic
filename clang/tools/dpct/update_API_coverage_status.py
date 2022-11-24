import argparse
import os
import sys
import warnings
import time
import string

output_file_suffix = '.md'

def get_user_answer(msg:str):
    msg += ' Are you sure? [y/n]'
    while(True):
        ans = str(input(msg))
        if ans=='y' or ans=='Y':
            return True
        elif ans == 'n' or ans == 'N':
            return False

def get_output_filename(path_str:str,filename:str,keep_filename:bool):
    if keep_filename is True:
        print("Output file is "+os.path.join(path_str,filename+output_file_suffix))
        return os.path.join(path_str,filename+output_file_suffix)
    new_file_name=filename+time.strftime('%Y-%m-%d %H:%M:%S')+output_file_suffix
    return get_output_filename(path_str, new_file_name,True)


def open_output_file(path_str:str,filename:str):
    keep_file = True
    filename_with_path = os.path.join(path_str,filename+output_file_suffix)
    if os.path.exists(filename_with_path) is True:
        keep_file = get_user_answer("Output file is existed, whether it can be overridden?")
    filename_with_path = get_output_filename(path_str,filename,keep_file)
    fout = open(filename_with_path,'w')
    return fout

def format_sing_line(API_list:list):
    return '|'+'{0: <60}'.format(API_list[0]) +'|'+'{0: ^10}'.format(API_list[1])+'|'+'{0: ^150}'.format(API_list[2])+'|\n'

def format_lib(lib_name:str,APIs_list:list):
    lib_str='## '+lib_name+'\n'
    lib_str += '| CUDA API | Migration support or not | Diagnostic message|\n'
    lib_str += '| :---- | :----: | :----: |\n'
    for API_list in APIs_list:
        lib_str += format_sing_line(API_list)
    lib_str += '\n'
    return lib_str



def update_remove(file_lib:str,fout):
    remove_help_msg = '_API calls from the original application, \
    which do not have functionally compatible SYCL API calls are removed \
        if the Intel® DPC++ Compatibility Tool determines \
            that it should not affect the program logic._'
    APIs_list=[]
    with open(file_lib,'r') as flib:
        for line in flib.readlines():
            img_file = line.strip()
            if(img_file=="" or img_file[0:5]!="ENTRY"):
                continue
            img_file=img_file.translate(str.maketrans('"()', ',,,'))
            img_list=img_file.split(',')
            API_list=[img_list[1]]
            API_list.append("YES")
            API_list.append(img_list[-3])
            APIs_list.append(API_list)
    lib_str = format_lib("removed API",APIs_list)
    if(fout.writable()):
        fout.write(lib_str)
        fout.write(remove_help_msg)
        return True
    else:
        warnings.warn("output file can not write, lost status information about removed")
        return False

def parse_macro_entry(line:str):
    line_list=line.split()
    API_list=[line_list[1]]
    if(line_list[3]=="true"):
        API_list.append("YES")
    elif (line_list[3]=="false"):
        API_list.append("NO")
    else:
        warnings.warn("internal error: can not tell whether API is supported")
        API_list.append("UNKNOW")
    if(line_list[-1][0:4]== 'DPCT'):
        API_list.append(line_list[-1])
    else:
        API_list.append('')
    return API_list

def parse_macro_entry_member_function(line:str):
    line_list=line.split()
    API_list=[line_list[1]+'::'+line_list[2]]
    if(line_list[4]=="true"):
        API_list.append("YES")
    elif (line_list[4]=="false"):
        API_list.append("NO")
    else:
        warnings.warn("internal error: can not tell whether API is supported")
        API_list.append("UNKNOW")
    if(line_list[-1][0:4]== 'DPCT'):
        API_list.append(line_list[-1])
    else:
        API_list.append('')
    return API_list



def update_lib(lib:str,file_lib:str,fout):
    APIs_list=[]
    with open(file_lib,'r') as flib:
        for line in flib.readlines():
            img_file = line.strip()
            if(img_file=="" or img_file[0:5]!="ENTRY"):
                continue
            img_file=img_file.translate(str.maketrans('"(),', '    '))
            if img_file[0:21]=='ENTRY_MEMBER_FUNCTION':
                API_list = parse_macro_entry_member_function(img_file)
            elif img_file[0:5]=="ENTRY":
                API_list = parse_macro_entry(img_file)
            APIs_list.append(API_list)
    lib_str = format_lib(lib,APIs_list)
    if(fout.writable()):
        fout.write(lib_str)
        return True
    else:
        warnings.warn("output file can not write, lost status information about "+lib)
        return False



def do_update(args):
    res = False
    DPCT_tools_path = os.path.dirname(__file__)
    DPCT_lib_path = os.path.join(DPCT_tools_path,'..','..','lib','DPCT')
    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_path = DPCT_tools_path
    if args.output_filename is not None:
        output_name = args.output_filename
    else:
        output_name = 'DPCT_API_COVERAGE'
    if os.path.exists(output_path) is False :
        warnings.warn("output path is not exist")
        return False
    fout = open_output_file(output_path,output_name)
    title_msg = '# The API Coverage Of DPC++ Compatibility Tools\n'
    fout.write(title_msg)
    lib_names = ['runtime and driver','CUB','cuBLAS','cuDNN','cuFFT','cuGRAPH','cuRAND','cuSOLVER','cuSPARSE','NCCL','nvJPEG','NVML','thrust','removed']
    # lib file name = APINames_$(libname).inc
    # runtime and drive is an exception
    for lib_name in lib_names:
        if lib_name == 'runtime and driver':
            lib_record_file = 'APINames.inc'
        else:
            lib_record_file = 'APINames'+'_'+lib_name+'.inc'
        lib_file = os.path.join(DPCT_lib_path,lib_record_file)
        if os.path.exists(lib_file) is False :
            warnings.warn(lib_file+" is not exist")
            continue
        if lib_name == "removed" :
            update_remove(lib_file,fout)
        else:
            update_lib(lib_name,lib_file,fout)
    fout.close()
    res = True
    return res





def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description="script to get the API coverage status in DPC++ Compatibility Tools",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--output-path",help="Set the path of the output file")
    parser.add_argument("--output-filename",help="Set the name of the output file")

    args = parser.parse_args()
    return do_update(args)


if __name__ == "__main__":
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)