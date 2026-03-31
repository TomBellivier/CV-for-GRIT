import json
import sys
import os

def format_json_file(input_path, output_path=None, indent=2):
    """
    Reads a compact JSON file and rewrites it with each element on its own line.

    Args:
        input_path  : Path to the source JSON file.
        output_path : Where to write the formatted result.
                      Defaults to overwriting the input file.
        indent      : Number of spaces used for indentation (default: 2).
    """
    if output_path is None:
        output_path = input_path.split(".json")[0] + "_formatted.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write("\n")  # trailing newline

    print(f"Formatted: {input_path} → {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python format_json.py <file1.json> [file2.json] ...")
        
        INPUT_DIR = "./annotations/formatting/not formatted/"
        OUTPUT_DIR = "./annotations/formatting/formatted/"
        for filename in os.listdir(INPUT_DIR):
            if filename.endswith(".json"):
                input_path = os.path.join(INPUT_DIR, filename)
                output_path = os.path.join(OUTPUT_DIR, filename)

                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                with open(input_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.write("\n")  # trailing newline
                
                os.rename(input_path, output_path)
                print(f"Formatted: {input_path} → {output_path}")

    else:
        for path in sys.argv[1:]:
            if not os.path.isfile(path):
                print(f"Skipping (not found): {path}")
                continue
            if not path.endswith(".json"):
                print(f"Skipping (not a .json file): {path}")
                continue
            try:
                format_json_file(path)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in {path}: {e}")


if __name__ == "__main__":
    main()