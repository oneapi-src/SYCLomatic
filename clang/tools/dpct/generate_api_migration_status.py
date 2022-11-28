import argparse
import os
import sys
import warnings
import time
import string

output_file_suffix = '.md'
keep_only_ask_once = False
keep_file = True

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
        return os.path.join(path_str,filename+output_file_suffix)
    new_file_name=filename+'_'+time.strftime('%Y-%m-%d %H:%M:%S')+output_file_suffix
    return get_output_filename(path_str, new_file_name,True)


def open_output_file(path_str:str,filename:str):
    global keep_only_ask_once
    global keep_file
    filename_with_path = os.path.join(path_str,filename+output_file_suffix)
    if os.path.exists(filename_with_path) is True :
        if keep_only_ask_once is False:
            keep_file = get_user_answer('output files is existed, whether it can be overridden?')
            keep_only_ask_once = True
        filename_with_path = get_output_filename(path_str,filename,keep_file)
    print("Output file is "+filename_with_path)
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


def parse_macro_entry(line:str):
    line_list=line.split()
    API_list=[line_list[1]]
    if(line_list[3]=="true"):
        API_list.append("YES")
    elif (line_list[3]=="false"):
        API_list.append("NO")
    else:
        warnings.warn("internal error: can not tell whether API is supported, please contact developer")
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



def update_lib(lib:str,file_lib:str,output_path:str):
    output_filename = lib+'_API_migration_status'
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
    fout = open_output_file(output_path,output_filename)
    if(fout.writable()):
        fout.write(lib_str)
        fout.close()
        return True
    else:
        warnings.warn("output file can not write, lost status information about "+lib)
        fout.close()
        return False



def do_update(args):
    res = True
    output_path = args.output_path
    SYCLomatic_path=args.SYCLomatic_path
    DPCT_lib_path = os.path.join(SYCLomatic_path,'clang','lib','DPCT')
    if os.path.exists(output_path) is False :
        warnings.warn("output path is not exist")
        return False
    lib_names = ['runtime_and_driver','CUB','cuBLAS','cuDNN','cuFFT','cuGRAPH','cuRAND','cuSOLVER','cuSPARSE','NCCL','nvJPEG','NVML','thrust']
    # lib file name = APINames_$(libname).inc
    # runtime and drive is an exception
    for lib_name in lib_names:
        if lib_name == 'runtime_and_driver':
            lib_record_file = 'APINames.inc'
        else:
            lib_record_file = 'APINames'+'_'+lib_name+'.inc'
        lib_file = os.path.join(DPCT_lib_path,lib_record_file)
        if os.path.exists(lib_file) is False :
            warnings.warn(lib_file+" is not exist")
            continue
        res= res&update_lib(lib_name,lib_file,output_path)
    return res





def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(__file__),
                                     description="A script to get the API migration status in DPC++ Compatibility Tools",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--output-path",help="Set the path of the output file",default=os.getcwd())
    parser.add_argument("--SYCLomatic-path",help="Set the path of the SYCLomatic",default=os.path.join(os.path.dirname(__file__),'..','..','..'))

    args = parser.parse_args()
    return do_update(args)


if __name__ == "__main__":
    ret = main()
    exit_code = 0 if ret else 1
    sys.exit(exit_code)