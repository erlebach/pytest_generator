import yaml
from generator_utils import sanitize_function_name

# from generator_utils import get_decoded_str
# from generator_utils import evaluate_answers


class TestConfig:
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def get(self, path, default=None):
        keys = path.split(".")
        result = self.config
        try:
            for key in keys:
                result = result[key]
            return result
        except KeyError:
            return default


class TestGenerator:
    def __init__(self, config_path, questions_path):
        self.config = TestConfig(config_path)
        self.questions_data = self.load_yaml_file(questions_path)
        self.test_code = ""
        self.types_config = self.config.get("types", {})

    def load_yaml_file(self, file_path):
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    def generate_tests(self):
        for question in self.questions_data["questions"]:
            for part in question["parts"]:
                self.generate_test_code(question, part)

    def generate_test_code(self, question, part):
        template = self.prepare_template(question, part)
        self.test_code += template

    def prepare_template(self, question, part):
        part_id = part["id"]
        test_type = part.get("type", "default_type")
        max_score = part.get("max_score", 10)
        fixture_name = self.questions_data.get("fixtures", {}).get("name", "")

        function_name = sanitize_function_name(
            f"test_{part_id.replace(' ', '_').replace(':', '_')}"
        )

        template = f"\n@max_score({max_score})\n@hide_errors('')\n"
        if fixture_name:
            template += f"def {function_name}({fixture_name}):\n"
        else:
            template += f"def {function_name}():\n"

        # Retrieve type-specific options
        type_options = self.types_config.get(test_type, {})

        # Generate setup lines based on type-specific options
        setup_lines = self.generate_type_setup(part, type_options)

        template += setup_lines
        template += "    assert True, 'Test Passed'\n"
        return template

    def generate_type_setup(self, part, type_options):
        setup_lines = ""
        options = type_options.get("options", [])

        # Handle tolerance options if applicable
        if "rel_tol" in options:
            rel_tol = part.get("rel_tol", self.config.get("defaults.rel_tol", 0.01))
            setup_lines += f"    rel_tol = {rel_tol}  # Relative tolerance\n"
        if "abs_tol" in options:
            abs_tol = part.get("abs_tol", self.config.get("defaults.abs_tol", 0.01))
            setup_lines += f"    abs_tol = {abs_tol}  # Absolute tolerance\n"

        # Handle partial scoring, indices inclusion/exclusion based on type
        if "partial_score" in options:
            setup_lines += "    partial_score = True  # Enable partial scoring\n"
        if "exclude_indices" in options:
            exclude_indices = part.get("exclude_indices", [])
            setup_lines += f"    exclude_indices = {exclude_indices}\n"
        if "include_indices" in options:
            include_indices = part.get("include_indices", [])
            setup_lines += f"    include_indices = {include_indices}\n"

        # Additional custom setups based on type
        if "local_namespaces" in options:
            namespaces = type_options["local_namespaces"]
            setup_lines += f"    # Setup namespaces: {namespaces}\n"

        return setup_lines

    def write_to_file(self, output_path):
        with open(output_path, "w") as file:
            file.write(self.test_code)

    def run(self):
        self.generate_tests()
        self.write_to_file("output_test_path.py")


# ------------------------------------------------
if __name__ == "__main__":
    test_generator = TestGenerator("generator_config.yaml", "part1.yaml")
    test_generator.run()
