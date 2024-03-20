#!/bin/bash

in="$1"
out="preprocessed_$1"
pre="preprocessed_$1_expand"

python test_utils.py -f "$in"

python yaml_expand.py --yaml "$out.yaml" -o "$pre.yaml"

./generate_structure.x $pre.yaml
./generate_answers.x $pre.yaml
