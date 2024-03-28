#!/bin/bash

in="$1"
out="preprocessed_$1"
pre="preprocessed_$1_expand"

# Generate preprocessed file "$out" (encode answers)
python test_utils.py -f "$in"

# Generated expanded yaml "$pre"
python yaml_expand.py --yaml "$out.yaml" -o "$pre.yaml" 

echo "===> answers"
python test_generator.py  --yaml "$pre.yaml" --simtype 'answers'
echo "===> structure"
python test_generator.py  --yaml "$pre.yaml" --simtype 'structure'
