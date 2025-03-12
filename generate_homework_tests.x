#!/bin/bash

in="$1"
out="preprocessed_$1"
pre="preprocessed_$1_expand"

# Generate preprocessed file "$out" (encode answers)
python -m pytest_generator.test_utils -f "$in"

# Generated expanded yaml "$pre"
python -m pytest_generator.yaml_expand --yaml "$out.yaml" -o "$pre.yaml" 

### # 
### # echo "===> answers"
python -m pytest_generator.test_generator  --yaml "$pre.yaml" --simtype 'answers'
### # echo "===> structure"
python -m pytest_generator.test_generator  --yaml "$pre.yaml" --simtype 'structure'
