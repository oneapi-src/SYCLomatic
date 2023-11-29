import os
import sys
import glob

def main():
    # work through all the files.
    dir_path = os.getcwd()
    print(dir_path)
    for (dir_path, dir_name, file_names) in os.walk(dir_path):
        for file_name in file_names:
            file_path = os.path.join(dir_path, file_name)
            expected_end = (".cu", ".cpp", ".cc", ".c")
            if (file_path.endswith(expected_end)):
                output = ""
                changed = False
                with open(file_path) as f:
                    for line in f.readlines():
                        if "FileCheck" in line:
                            options = line.split(' ')
                            for option in options:
                                if "%T" in option and "cpp" in option:
                                  changed = True
                                  option = option.strip('\n')
                                  out_obj = option.replace(".cpp",".o")
                                  cmd = "// RUN: %if build_lit %{icpx -c -fsycl "+option + " -o " + out_obj + " %}"
                                  line += cmd + "\n"
                        output += line
                if changed:
                    with open(file_path, 'w') as f:
                        f.write(output)
    # parse get the Filecheck input file.
    # generated line
    
if __name__ == "__main__":
    main()