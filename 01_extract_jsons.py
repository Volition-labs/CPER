import os
import json
import glob

def load_json_file(filepath):
    """
    Attempt to load a JSON object from a file.
    If the JSON is complete but wrapped in triple backticks (```), they will be removed.
    If the JSON is truncated or cannot be loaded, the file is skipped and a note is printed.

    Returns:
        A JSON object (dictionary) if successfully loaded, or None if failed.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Remove triple backticks if present at the beginning and/or end.
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        # Attempt to load the cleaned JSON string.
        data = json.loads(content)
        return data
    except json.JSONDecodeError as e:
        # Log the error and note that the JSON might be truncated.
        print(f"JSON decoding error in {filepath}: {e}. Possibly truncated JSON. Skipping.")
        return None

def load_json_files(filepaths):
    """
    Given a list of file paths, attempt to load each JSON file using load_json_file.
    Returns a list of successfully loaded JSON objects.
    """
    data_list = []
    for filepath in filepaths:
        data = load_json_file(filepath)
        if data is not None:
            data_list.append(data)
    return data_list


if __name__ == '__main__':
    # Replace this with the path to your JSON files directory.
    directory_path = "/home/stirunag/Downloads/Downloaded_EPD_JSON-20250327T001140Z-001/Downloaded_EPD_JSON"  # Replace with your JSON directory path
    all_files = glob.glob(os.path.join(directory_path, "*.json"))
    data = load_json_files(all_files)
    print(f"Loaded {len(data)} JSON objects.")

    # Assume 'data' is your processed list of JSON objects
    output_file = 'data/processed_json_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Data saved to {output_file}")


# # Example usage:
#
# all_json_data = load_json_files(directory_path)
#
# # Preprocess each JSON file and store the result
# processed_data = [preprocess_epd_json(epd) for epd in all_json_data]
#
# # For example, let's inspect the first processed JSON
# import pprint
#
# pprint.pprint(processed_data[0])
