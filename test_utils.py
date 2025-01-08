import yaml
import base64
import json
import argparse
import sys

def encode_answer(answer):
    # Convert any non-string answer to string before encoding
    answer_str = str(answer)
    answer_bytes = answer_str.encode('utf-8')  # Convert to bytes
    encoded_bytes = base64.b64encode(answer_bytes)
    encoded_str = encoded_bytes.decode('utf-8')  # Convert bytes back to string
    return encoded_str


def preprocess_yaml(input_file, output_file):
    print("input file: ", input_file)
    print("output file: ", output_file)
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)

    if data is None:
        sys.exit(f"Error: Empty or incorrect {input_file}")
    
    for question in data.get('questions', []):
        parts = question.get('parts', [])
        for i, part in enumerate(parts):
            # trick using eval (2025-01-07)
            # if part == 10, eval(str(part)) returns 10
            # if part == 'np.zeros(3)', eval(str(part)) returns np.zeros(3)
            # Encode all answers
            #part['answer'] = encode_answer(part['answer'])
            if 's_answers' in part and isinstance(part['s_answers'], list):
                answers = part['s_answers']
                for j, answ in enumerate(answers):
                    part['s_answers'][j] = encode_data(answ)
            if 'i_answers' in part and isinstance(part['i_answers'], list):
                #print(f"{part['i_answers']=}")
                answers = part['i_answers']
                for j, answ in enumerate(answers):
                    part['i_answers'][j] = encode_data(answ)
            if 's_answer' in part:
                # print("==> part['s_answer'] = ", part['s_answer'])
                part['s_answer'] = encode_data(part['s_answer'])
            if 'i_answer' in part:
                part['i_answer'] = encode_data(part['i_answer'])
            #print(f"{len(parts)=}, {i=}")
            parts[i] = part
            #print("part= ", part)
        question['parts'] = parts
    
    with open(output_file, 'w') as file:
        yaml.dump(data, file, sort_keys=False)


# Example: Encoding with type hints
def encode_data(data):
    # Why are np.ndarray strings? 
    print("==> encode_data: ", data)
    if isinstance(data, set):
        encoded = json.dumps({"type": "set", "data": list(data)})
    else:
        #print("data= ", data)
        encoded = json.dumps(data)  # Directly use JSON for other types
    return base64.b64encode(encoded.encode()).decode()

# Example: Decoding with type handling
def decode_data(encoded):
    decoded = json.loads(base64.b64decode(encoded).decode())
    # Not clear why the next two lines are needed
    if isinstance(decoded, dict) and decoded.get("type") == "set":
        return set(decoded["data"])
    if isinstance(decoded, str) and decoded[0:3] == 'set':
        return eval(decoded)
    return decoded



#----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script and what it does.")

    # Add the arguments
    parser.add_argument("-f", "--filename", type=str, help="The filename to process", required=True)

    # Parse the arguments
    args = parser.parse_args()

    # Usage (do not include yaml extension)
    input_file = args.filename  + ".yaml"
    output_file = f"preprocessed_{input_file}"
    preprocess_yaml(input_file, output_file)
