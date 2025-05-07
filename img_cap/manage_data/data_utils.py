import csv
import ast # For safely evaluating string representations of lists
import os

def convert_csv_to_txt(csv_filepath, txt_filepath):
    """
    Converts a CSV file with image annotations to a TXT file
    in the format: filename#index caption.

    Args:
        csv_filepath (str): The path to the input CSV file.
        txt_filepath (str): The path to the output TXT file.
    """
    try:
        with open(csv_filepath, 'r', encoding='utf-8') as csvfile, \
             open(txt_filepath, 'w', encoding='utf-8') as txtfile:

            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)  # Skip the header row

            # Get column indices based on header names for robustness
            try:
                raw_col_idx = header.index('raw')
                filename_col_idx = header.index('filename')
            except ValueError:
                print("Error: CSV file must contain 'raw' and 'filename' columns.")
                return

            print(f"Starting conversion from {csv_filepath} to {txt_filepath}...")

            for row_num, row in enumerate(csv_reader):
                if not row:  # Skip empty rows
                    continue
                try:
                    filename = row[filename_col_idx]
                    raw_captions_str = row[raw_col_idx]

                    # Safely evaluate the string representation of the list of captions
                    captions_list = ast.literal_eval(raw_captions_str)

                    if not isinstance(captions_list, list):
                        print(f"Warning: Row {row_num + 2}: 'raw' column is not a list: {raw_captions_str}")
                        continue

                    for i, caption in enumerate(captions_list):
                        if not isinstance(caption, str):
                            print(f"Warning: Row {row_num + 2}, Caption {i}: Caption is not a string: {caption}")
                            continue
                        # Ensure the caption ends with a period if it doesn't already have one or other punctuation
                        # and remove leading/trailing whitespace.
                        cleaned_caption = caption.strip()
                        if cleaned_caption and not cleaned_caption.endswith(('.', '!', '?')):
                            cleaned_caption += " ." # Add space before period for consistency with caption.txt
                        else:
                            # If it ends with punctuation, ensure there's a space before it if it's a period.
                            # This part is a bit tricky if various punctuations are possible.
                            # For simplicity, we'll just ensure a space if it ends with a period.
                            if cleaned_caption.endswith('.') and not cleaned_caption.endswith(' .'):
                                cleaned_caption = cleaned_caption[:-1] + " ."


                        txtfile.write(f"{filename}#{i}\t{cleaned_caption}\n")
                except IndexError:
                    print(f"Warning: Row {row_num + 2} is malformed or has too few columns: {row}")
                except (SyntaxError, ValueError) as e:
                    print(f"Warning: Row {row_num + 2}: Could not parse 'raw' captions: {raw_captions_str}. Error: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred at row {row_num + 2}: {e}")

            print("Conversion completed successfully.")

    except FileNotFoundError:
        print(f"Error: The file {csv_filepath} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def split_filenames_by_dataset_split(csv_filepath, output_dir):
    """
    Reads a CSV file and splits filenames into train, val, and test
    based on the 'split' column.

    Args:
        csv_filepath (str): The path to the input CSV file.
        output_dir (str): The directory where the output TXT files will be saved.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Define output file paths
    train_file_path = os.path.join(output_dir, 'train_imgs.txt')
    val_file_path = os.path.join(output_dir, 'val_imgs.txt')
    test_file_path = os.path.join(output_dir, 'test_imgs.txt')

    # Using sets to store filenames to avoid duplicates if any image_id appears multiple times per split
    # (though typically each image_id is unique per split in such datasets)
    train_filenames = set()
    val_filenames = set()
    test_filenames = set()

    try:
        with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)  # Skip the header row

            # Get column indices
            try:
                filename_col_idx = header.index('filename')
                split_col_idx = header.index('split')
            except ValueError:
                print("Error: CSV file must contain 'filename' and 'split' columns.")
                return

            print(f"Processing {csv_filepath}...")
            for row_num, row in enumerate(csv_reader):
                if not row: # Skip empty rows
                    continue
                try:
                    filename = row[filename_col_idx]
                    split_type = row[split_col_idx].strip().lower() # Normalize to lowercase

                    if split_type == 'train':
                        train_filenames.add(filename)
                    elif split_type == 'val':
                        val_filenames.add(filename)
                    elif split_type == 'test':
                        test_filenames.add(filename)
                    else:
                        print(f"Warning: Row {row_num + 2}: Unknown split type '{row[split_col_idx]}' for filename {filename}")
                except IndexError:
                    print(f"Warning: Row {row_num + 2} is malformed or has too few columns: {row}")
                except Exception as e:
                    print(f"An unexpected error occurred at row {row_num + 2}: {e}")


        # Write to output files
        with open(train_file_path, 'w', encoding='utf-8') as f_train:
            for fname in sorted(list(train_filenames)): # Sort for consistent order
                f_train.write(f"{fname}\n")
        print(f"Saved {len(train_filenames)} filenames to {train_file_path}")

        with open(val_file_path, 'w', encoding='utf-8') as f_val:
            for fname in sorted(list(val_filenames)):
                f_val.write(f"{fname}\n")
        print(f"Saved {len(val_filenames)} filenames to {val_file_path}")

        with open(test_file_path, 'w', encoding='utf-8') as f_test:
            for fname in sorted(list(test_filenames)):
                f_test.write(f"{fname}\n")
        print(f"Saved {len(test_filenames)} filenames to {test_file_path}")

        print("Processing completed successfully.")

    except FileNotFoundError:
        print(f"Error: The file {csv_filepath} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Define your input and output file paths
    # Make sure these paths are correct for your system
    input_csv_file = './data/flickr30k/flickr_annotations_30k.csv'
    output_txt_file = './data/flickr30k/caption.txt' # Or your desired output path
    output_directory = './data/flickr30k/' # Or your desired output directory
    # convert_csv_to_txt(input_csv_file, output_txt_file)
    split_filenames_by_dataset_split(input_csv_file, output_directory)