// RUN: rm -rf %T && mkdir -p %T
// RUN: cd %T
// RUN: cp %S/python_setup_input.py ./python_setup_input.py
// RUN: mkdir -p subdir
// RUN: dpct -in-root ./ -out-root out  ./python_setup_input.py --migrate-build-script-only
// RUN: echo "begin" > %T/diff.txt
// RUN: diff --strip-trailing-cr %S/python_setup_expected.py %T/out/python_setup_input.py >> %T/diff.txt
// RUN: echo "end" >> %T/diff.txt
