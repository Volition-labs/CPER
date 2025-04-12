import os
import json
import glob
import csv
from collections import Counter
from tqdm import tqdm

def get_impact_info(epd_json, impact_filter='global warming'):
    impacts = epd_json.get("epd_impacts", [])
    for impact in impacts:
        if impact_filter.lower() in impact.get("impact_category", "").lower():
            return impact
    return None

def is_placeholder_entry(entry):
    return (
        entry.get('product_names') == ['Product Name'] or
        entry.get('product_ids') == ['Product ID'] or
        entry.get('product_description') == ['Product Description']
    )

def has_valid_impact(impact_info):
    keys_to_check = ['A1', 'A2', 'A3', 'A4', 'A5', 'A1_A3_total']
    return any(impact_info.get(key) not in [0, None] for key in keys_to_check)

def clean_json(entry, impact_info):
    entry['epd_impacts'] = [impact_info]
    return entry

def process_json_directory(directory_path, known_products=None, status_log=None):
    all_files = glob.glob(os.path.join(directory_path, "*.json"))
    processed_data = []

    for filepath in tqdm(all_files, desc=f"Processing {directory_path}"):
        file_name = os.path.basename(filepath)
        folder_name = os.path.basename(directory_path)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                status_log.append([folder_name, file_name, 'json_validation_error'])
                continue

            if is_placeholder_entry(data):
                status_log.append([folder_name, file_name, 'null_values'])
                continue

            impact_info = get_impact_info(data)
            if not impact_info or not has_valid_impact(impact_info):
                status_log.append([folder_name, file_name, 'null_values'])
                continue

            product_key = (
                tuple(data.get('product_names', [])),
                tuple(data.get('product_ids', [])),
                tuple(data.get('product_description', []))
            )

            if known_products is not None and product_key in known_products:
                status_log.append([folder_name, file_name, 'duplicate_values'])
                continue

            processed_data.append(clean_json(data, impact_info))
            if known_products is not None:
                known_products.add(product_key)

            status_log.append([folder_name, file_name, 'processed'])

        except Exception as e:
            status_log.append([folder_name, file_name, 'error'])
            print(f"Failed to load {filepath}: {e}")
            continue

    return processed_data

if __name__ == '__main__':
    directory_path_0 = "/home/stirunag/Downloads/Complete_EPD_JSON-20250411T185504Z-001/Complete_EPD_JSON"
    directory_path_1 = "/home/stirunag/Downloads/Downloaded_EPD_JSON-20250327T001140Z-001/Downloaded_EPD_JSON"

    known_products = set()
    status_log = []

    data_0 = process_json_directory(directory_path_0, known_products, status_log)
    print(f"Directory 0: {len(data_0)} valid JSONs")

    data_1 = process_json_directory(directory_path_1, known_products, status_log)
    print(f"Directory 1: {len(data_1)} new non-duplicate JSONs")

    final_data = data_0 + data_1
    output_file = 'data/revised_processed_json_data_v01.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"Total saved: {len(final_data)} JSONs -> {output_file}")

    # Save status log to CSV
    log_file = 'data/processing_log.csv'
    with open(log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['folder_name', 'file_name', 'status'])
        writer.writerows(status_log)

    print(f"Log saved: {log_file}")
