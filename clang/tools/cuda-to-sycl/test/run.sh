#!/bin/bash

set -o errexit
set -o nounset

TESTDIR="$(dirname "$0")"
TESTDIR="$(cd "$TESTDIR"; pwd)"
NFAIL=0

echodo() {
    echo + "$@"
    "$@"
}

for tst in "$TESTDIR"/*; do
    if ! [ -d "$tst" ]; then
        continue
    fi

    echodo cp "$tst"/input.cu "$tst"/output-exp.cc
    echodo cuda-to-sycl "$tst"/output-exp.cc -- \
           -x cuda --cuda-path="$CU2SYCL_CUDA_PATH" --cuda-host-only
    if echodo diff "$tst"/output-ref.cc "$tst"/output-exp.cc; then
        # No need to keep output file if it matches the reference one.
        echodo rm "$tst"/output-exp.cc
    else
        NFAIL=$((NFAIL + 1))
    fi
done

if [ $NFAIL -eq 0 ]; then
    echo "SUCCESS"
    exit 0
else
    echo "$NFAIL tests FAILED"
    exit 1
fi
