#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo Usage: $0 case
    exit 1
fi

cd ..; ./read_solution.py $1; cd soar
