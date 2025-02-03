#!/bin/bash

# Check for missing argument
if [ $# -eq 0 ]; then
    echo "Error: Argument missing. Please provide a file name without the .yaml extension."
    exit 1
fi

# Remove .yaml extension if present
filename=${1%.yaml}

# $1 is the yaml file or yaml files (without the extension)
./generate_homework_tests.x $filename
