import yaml

#with open("all_questions_hw3.yaml", "r") as f:
#with open("part1.yaml", "r") as f:
with open("type_handlers.yaml", "r") as f:
    data = yaml.safe_load(f)

print(data)
