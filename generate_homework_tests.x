#!/bin/bash

in="$1"
out="preprocessed_$1"
pre="preprocessed_$1_expand"

# Generate preprocessed file (encode answers)
python test_utils.py -f "$in"

# Generated expanded yaml 
python yaml_expand.py --yaml "$out.yaml" -o "$pre.yaml" 

python test_answers_generator.py  --yaml "$pre.yaml" --simtype 'answers'
python test_answers_generator.py  --yaml "$pre.yaml" --simtype 'structure'
