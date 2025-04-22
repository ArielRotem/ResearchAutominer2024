import pandas as pd
from collections import defaultdict
import re


# Custom sort function
def custom_sort(item):
    # 'Other' will have the highest priority
    return (item != "Uncategorized", item)

def excel_column_to_index(col):
    """
    Convert an Excel-style column label to a zero-based column index.
    Example: 'A' -> 0, 'B' -> 1, 'Z' -> 25, 'AA' -> 26, 'BC' -> 54
    """
    index = 0
    for c in col:
        index = index * 26 + (ord(c.upper()) - ord('A') + 1)
    return index - 1  # convert to zero-based index

def column_name_to_index(data, column_name):
    """
    Given a DataFrame and a column name, return the zero-based index of the column.
    Raises a KeyError if the column is not found.
    """
    # Use pandas' .get_loc() to find the index of the column name
    if column_name in data.columns:
        return data.columns.get_loc(column_name)
    else:
        raise KeyError(f"Column '{column_name}' not found in the DataFrame.")

def update_dataframe(originalData, col1, words1, logical_op, col2, words2, col3, step_size, num_steps, col3_empty, result_column_name, return_values=True, unique=False, dictionary=None, limitResults=None):    
    data = originalData.copy()
    # Prepare column indices
    col1_index = column_name_to_index(data, col1) if col1 else None
    col2_index = column_name_to_index(data, col2) if col2 else None
    col3_index = column_name_to_index(data, col3) if col3 else None

    # Function to apply on each row
    def evaluate_row(row):
        concatenated_results = []  # List to collect valid results

        for step in range(num_steps):
            offset = step * step_size
            current_col1_index = col1_index + offset if col1_index is not None else None
            current_col2_index = col2_index + offset if col2_index is not None else None
            current_col3_index = col3_index + offset if col3_index is not None else None

            # Ensure column indexes are within the bounds of the row
            if (current_col1_index and current_col1_index >= len(row)) or (current_col2_index and current_col2_index >= len(row)) or (current_col3_index and current_col3_index >= len(row)):
                continue

            # Convert the data to string for regex matching
            data_col1 = str(row.iloc[current_col1_index]) if current_col1_index is not None else ''
            data_col2 = str(row.iloc[current_col2_index]) if current_col2_index is not None else ''
            data_col3 = str(row.iloc[current_col3_index]) if current_col3_index is not None else ''
            
            # Check conditions
            cond1 = any(word in data_col1 for word in words1) if col1 and words1 else True
            cond2 = any(word in data_col2 for word in words2) if words2 else True
            cond3 = True if col3_index is None else (bool(data_col3) if not col3_empty else not data_col3)
            
            #if data_col2 != '':
                #print(f"data: {data_col2}, words: {words2}, cond: {cond2}")

            # Adjust logic operation based on whether col1 is effectively in use
            if col1 and words1:
                condition = (cond1 and cond2) if logical_op == 'AND' else (cond1 or cond2)
            else:
                condition = cond1 and cond2

            # Collect results if conditions are met
            if condition and cond3:
                if return_values:
                    concatenated_results.extend([item for item in data_col3.split(';') if item.strip()])

                else:
                    concatenated_results.append(True)  # Collect 'True' values for existence check
                    break # one true is enough

        # Final assignment based on operation mode
        if return_values:
            if dictionary:
                # Replace values based on dictionary with default value "Other" if key not found
                concatenated_results = [dictionary.get(item, "Uncategorized") for item in concatenated_results]

            if unique:
                # Deduplicate by converting to set and back to list
                concatenated_results = list(set(concatenated_results))
            
            if limitResults is not None and len(concatenated_results) > limitResults:
                # Handle the limitResults parameter
                concatenated_results = concatenated_results[:limitResults]
            
            concatenated_results = sorted(concatenated_results, key=lambda x: ('' if x == 'Uncategorized' else x))
            return ', '.join(map(str, concatenated_results)) if concatenated_results else ''

        else:
            return int(any(concatenated_results)) ## swap from true to: true=1, false=0


    # Apply the function and assign results to the new result column
    results = data.apply(evaluate_row, axis=1)
    data.loc[:, result_column_name] = results
    print(f"done working on new column: {result_column_name}")
    return data

def remove_rows_below_threshold(data, column_name, threshold):
    return data[data.iloc[:, column_name_to_index(data, column_name)] >= threshold]
    
def remove_rows_above_threshold(data, column_name, threshold):
    return data[data.iloc[:, column_name_to_index(data, column_name)] <= threshold]

def remove_rows_if_contains(data, column_name, words):
    return data[~data.iloc[:, column_name_to_index(data, column_name)].astype(str).str.lower().apply(lambda x: any(word.lower() in x for word in words))]


def update_column_with_values(data, column_name, words_dict, default_value="0", empty_value=None):
    column_index = column_name_to_index(data, column_name)

    def replace_value(cell_value):
        cell_value_lower = cell_value.lower()  # Lowercase the cell value for case-insensitive comparison
        if cell_value == "":  # Check if the cell is empty
            return empty_value if empty_value is not None else cell_value
        for key, words in words_dict.items():
            if any(word.lower() in cell_value_lower for word in words):
                return key
        if cell_value != "":
            print(f"Failed to translate dictionary value! Col:{column_name} Value:{cell_value}")
            return default_value  # Use the specified default value if no matches found
        return ""

    data.iloc[:, column_index] = data.iloc[:, column_index].astype(str).apply(replace_value)

def clear_negative_values(data, column_name):
    col_index = column_name_to_index(data, column_name)
    # Convert column values to numbers in a temporary series, treating errors and empty strings
    temp_series = pd.to_numeric(data.iloc[:, col_index], errors='coerce')
    # Mask to find negative values in the temporary series
    negative_mask = temp_series < 0
    # Count negative values
    count_negative = negative_mask.sum()
    # Print negative values
    # print("Negative values before clearing:")
    # print(temp_series[negative_mask])
    # Clear negative values in the original data
    data.loc[negative_mask, data.columns[col_index]] = ''
    # Return the count of cleared cells
    return count_negative

## Alias functions    
def containswords_andor_containswords_and_nonempty_result_values(data, col1, words1, logical_op, col2, words2, col3, step_size, num_steps, result_column_name, unique=True, dictionary=None, limitResults=None):
    return update_dataframe(data, col1, words1, logical_op, col2, words2, col3, step_size, num_steps, False, result_column_name, return_values=True, unique=unique, dictionary=dictionary, limitResults=limitResults)

def containswords_and_nonempty_result_values(data, col2, words2, col3, step_size, num_steps, result_column_name, unique=True, dictionary=None, limitResults=None):
    return update_dataframe(data, '', '', '', col2, words2, col3, step_size, num_steps, False, result_column_name, return_values=True, unique=unique, dictionary=dictionary, limitResults=limitResults)

def containswords_andor_containswords_and_nonempty_result_exists(data, col1, words1, logical_op, col2, words2, col3, step_size, num_steps, result_column_name):
    return update_dataframe(data, col1, words1, logical_op, col2, words2, col3, step_size, num_steps, False, result_column_name, return_values=False, unique=False, dictionary=None)

def containswords_andor_containswords_result_exists(data, col1, words1, logical_op, col2, words2, step_size, num_steps, result_column_name):
    return update_dataframe(data, col1, words1, logical_op, col2, words2, '', step_size, num_steps, False, result_column_name, return_values=False, unique=False, dictionary=None)

def containswords_result_exists(data, col2, words2, step_size, num_steps, result_column_name):
    return update_dataframe(data, '', '', '', col2, words2, '', step_size, num_steps, True, result_column_name, return_values=False, unique=False, dictionary=None)

def clear_strings_multiple_columns(data, column_names, words, indicator=-1):
    words_lower = [word.lower() for word in words]
    
    for column_name in column_names:
        # Convert column name to index
        col_index = column_name_to_index(data, column_name)
        # Determine adjacent column index based on indicator
        adj_col_index = col_index + indicator if indicator != 0 else col_index
        
        # Ensure adjacent column index is within bounds
        if not (0 <= adj_col_index < data.shape[1]):
            print(f"Adjacent column index out of bounds for column {column_name}.")
            continue
        
        # Create mask for cells containing any of the words
        mask = data.iloc[:, col_index].astype(str).str.lower().apply(lambda x: any(word in x for word in words_lower))
        
        # Clear values in the main column and the adjacent column based on the mask
        data.iloc[mask, col_index] = ""
        if indicator != 0:  # Only clear adjacent column if indicator is not 0
            data.iloc[mask, adj_col_index] = ""
    
    return data

def custom_logic_operation(data, col1_name, col2_name, result_col_name):
    # Create a copy of the DataFrame to avoid modifying the original
    data_copy = data.copy()
    
    # Convert column names to indices
    col1_idx = column_name_to_index(data_copy, col1_name)
    col2_idx = column_name_to_index(data_copy, col2_name)
    
    # Apply the custom logic to each row
    def apply_logic(row):
        col1 = row[col1_idx]
        col2 = row[col2_idx]
        
        # Check if both are empty
        if pd.isna(col1) and pd.isna(col2):
            return None
        
        # Apply the rules
        if col1 == "1" or col2 == "1":
            return 1
        if col1 == "0" or col2 == "0":
            return 0
        if col1 == "2" or col2 == "2":
            return 2
        
        # Default to empty if no conditions are met
        return None

    # Apply the logic to each row and set the result in the new column
    data_copy[result_col_name] = data.apply(apply_logic, axis=1)
    
    return data_copy

def concat_unique_values(data, column_names, new_column_name, limitResults=None):
    """
    Creates a new column in the DataFrame by concatenating unique values from specified columns,
    limiting the number of concatenated results if specified.

    Args:
    data (pd.DataFrame): The DataFrame to modify.
    column_letters (list of str): List of Excel-style column letters of the columns to concatenate.
    new_column_name (str): Name of the new column to create.
    limitResults (int, optional): Maximum number of results to include in the new column for each row.

    Returns:
    pd.DataFrame: A copy of the DataFrame with the new column added.
    """
    data_copy = data.copy()
    
    # Convert column names to indices
    column_indices = [column_name_to_index(data_copy, name) for name in column_names]
    
    # Function to process each row
    def process_row(row):
        # Collect values from specified columns using .iloc for proper indexing
        values = [row.iloc[idx] for idx in column_indices if pd.notna(row.iloc[idx]) and row.iloc[idx] != '']
        
        # Remove duplicates
        unique_values = list(set(values))
        
        # If a limit is set, apply it before sorting
        if limitResults is not None:
            unique_values = unique_values[:limitResults]
        
        # Sort the values to ensure consistent order
        sorted_values = sorted(unique_values)
        
        # Join values with a comma
        return ', '.join(sorted_values)
    
    # Apply the function to each row and assign to the new column
    data_copy[new_column_name] = data_copy.apply(process_row, axis=1)
    
    return data_copy

def remove_columns(data, column_names):
    """
    Remove specified columns from a DataFrame based on Excel-style column names or ranges.

    Args:
    data (pd.DataFrame): The DataFrame from which to remove columns.
    column_letters (list of str): List of Excel-style column letters or ranges to be removed.

    Returns:
    pd.DataFrame: The DataFrame with specified columns removed.
    """
    # This list will store all the column names to remove
    columns_to_remove = []

    # Process each specified column name or range
    for col in column_names:
        if '~' in col:
            # Handle range: split on '~', convert start and end to indices, and get all columns in that range
            start_col, end_col = col.split('~')
            start_index = column_name_to_index(data, start_col)
            end_index = column_name_to_index(data, end_col)
            # Append all columns in this range to the list
            columns_to_remove.extend(data.columns[start_index:end_index+1])
        else:
            # Convert single column name to index and append the column name to the list
            index = column_name_to_index(data, col)
            columns_to_remove.append(data.columns[index])

    # Remove the collected columns from the DataFrame
    data.drop(columns=columns_to_remove, inplace=True)
    return data

def clear_values_based_on_reference(data, target_column_name, reference_column_name, reference_value):
    """
    Clears values in the target column based on a condition met in the reference column.

    Args:
    data (pd.DataFrame): The DataFrame to modify.
    target_column_letter (str): Excel-style column letter of the target column to clear.
    reference_column_letter (str): Excel-style column letter of the reference column to check.
    reference_value (any): The value to check against in the reference column.

    Returns:
    pd.DataFrame: The DataFrame with modified values.
    """
    # Convert column names to indices
    target_index = column_name_to_index(data, target_column_name)
    reference_index = column_name_to_index(data, reference_column_name)

    # Define the function to clear values based on reference column condition
    def clear_value(row):
        # Check if the reference column value matches the reference value
        if row.iloc[reference_index] == reference_value:
            return ""  # Clear the target column value
        else:
            return row.iloc[target_index]  # Keep the original value

    # Apply the function to each row
    data.iloc[:, target_index] = data.apply(clear_value, axis=1)
   
    return data



def is_empty(data, column_name, new_column_name, value_empty=1, value_not_empty=0 ):
    """
    Checks if each cell in the specified column is not empty.
    
    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    column_index = column_name_to_index(data, column_name)
    data[new_column_name] = data.iloc[:, column_index].apply(lambda x: value_not_empty if pd.notna(x) and x != '' else value_empty)
    return data

def filter_numbers(data, column_name, lowerThan=None, higherThan=None, emptyOk=True):
    """
    Filters numbers in a column based on specified thresholds.
    
    Args:
    data (pd.DataFrame): The DataFrame to modify.
    column_letter (str): Excel-style column letter to filter.
    lowerThan (float): Lower threshold. Numbers below this will be removed.
    higherThan (float): Upper threshold. Numbers above this will be removed.
    
    Returns:
    pd.DataFrame: The DataFrame with filtered values.
    """
    column_index = column_name_to_index(data, column_name)
    def filter_val(x):
        try:
            if (emptyOk and (x is None or x == "")):
                return x
            num = float(x)
            if (lowerThan is not None and num < lowerThan) or (higherThan is not None and num > higherThan):
                return ''
            else:
                return x
        except ValueError:
            print("err in filter_val: ", x)
            return x
    data.iloc[:, column_index] = data.iloc[:, column_index].apply(filter_val)
    return data

def flip_sign(data, column_name):
    """
    Flips the sign of numeric values in the specified column.
    
    Args:
    data (pd.DataFrame): The DataFrame to modify.
    column_letter (str): Excel-style column letter of the column to modify.
    
    Returns:
    pd.DataFrame: The DataFrame with modified values.
    """
    column_index = column_name_to_index(data, column_name)
    def flip(x):
        try:
            return -float(x)
        except ValueError:
            return x
    data.iloc[:, column_index] = data.iloc[:, column_index].apply(flip)
    return data
    
def multiply_by_number(data, column_name, multiplier):
    """
    Multiplies the values in the specified column by the multiplier.
    
    """
    column_index = column_name_to_index(data, column_name)
    def multiply(x):
        try:
            return float(x)*multiplier
        except ValueError:
            return x
    data.iloc[:, column_index] = data.iloc[:, column_index].apply(multiply)
    return data    

def cutoff_number(data, column_name, new_column_name, cutoff, above=1, below=0, empty_value=None):
    """
    Creates a new column in the DataFrame to indicate how cell values in a specified column 
    compare against a given numeric cutoff. Values above the cutoff return 'above', values below
    return 'below', and non-numeric or empty cells return 'empty_value'.

    Args:
    data (pd.DataFrame): The DataFrame to modify.
    column_letter (str): Excel-style column letter of the column to check.
    new_column_name (str): Name of the new column to create.
    cutoff (float): Cutoff value to compare against.
    above (int): Value to assign to the new column if the cell value is above the cutoff.
    below (int): Value to assign to the new column if the cell value is below the cutoff.
    empty_value (any): Value to assign to the new column if the original cell is empty or non-numeric.

    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    column_index = column_name_to_index(data, column_name)
    def check_cutoff(x):
        try:
            if float(x) >= cutoff:
                return above
            else:
                return below
        except ValueError:
            return empty_value

    data[new_column_name] = data.iloc[:, column_index].apply(check_cutoff)
    return data

def compare_values(data, column_name, new_column_name, target_value, match_return, no_match_return):
    """
    Creates a new column in a DataFrame to indicate whether cell values in a specified column 
    match a given value, considering type coercion to handle both numeric and string comparisons.
    Returns 'match_return' if they match, 'no_match_return' otherwise.

    Args:
    data (pd.DataFrame): The DataFrame to modify.
    column_letter (str): Excel-style column letter of the column to check.
    new_column_name (str): Name of the new column to create.
    target_value (any): Value to compare against the cell values. If it's a string that can be a number, comparisons will consider numeric equivalence.
    match_return (any): Value to return in the new column if the cell matches the target_value.
    no_match_return (any): Value to return in the new column if the cell does not match the target_value.

    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    column_index = column_name_to_index(data, column_name)
    
    def check_match(x):
        try:
            # Attempt to convert both x and target_value to floats for comparison
            if float(x) == float(target_value):
                return match_return
        except ValueError:
            # If conversion fails, fall back to string comparison
            if str(x) == str(target_value):
                return match_return
        return no_match_return

    data[new_column_name] = data.iloc[:, column_index].apply(check_match)
    return data

def combine_columns(data, column_names, new_column_name, delimiter=" "):
    """
    Combines multiple columns into a new column in a DataFrame, using a specified delimiter.

    Args:
    data (pd.DataFrame): The DataFrame to modify.
    column_letters (list of str): List of Excel-style column letters of the columns to combine.
    new_column_name (str): Name of the new column to create.
    delimiter (str): Delimiter to use for combining the column values.

    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    # Convert column letters to DataFrame column indices
    column_indices = [column_name_to_index(data, name) for name in column_names]
    
    # Function to process each row
    def combine_row(row):
        # Collect values from specified columns using .iloc for proper indexing, convert to string to ensure proper concatenation
        values = [str(row.iloc[idx]) for idx in column_indices]
        # Filter out empty strings to avoid unnecessary delimiters
        filtered_values = [value for value in values if value and value != 'nan']
        # Remove duplicates
        filtered_values = list(set(filtered_values))
        
        # Additional filter to remove '0' if there are other numbers
        if '0' in filtered_values and len(filtered_values) > 1:
            filtered_values.remove('0')
        
        # Join values with the specified delimiter
        return delimiter.join(filtered_values)

    # Apply the function to each row and assign to the new column
    data[new_column_name] = data.apply(combine_row, axis=1)
    
    return data

def remove_contaminant_and_count(data, column_name, new_column_name, delimiter=',', default_value='', contaminant=''):
    """
    Process each cell in the specified column by splitting it by the delimiter, removing the contaminant,
    and replacing the cell with either the default value or a count of the remaining elements.

    Args:
    data (pd.DataFrame): The DataFrame to modify.
    column_name (str): The name of the column to operate on.
    delimiter (str): The delimiter to split the cell by (default is comma ',').
    default_value (str): The value to use if the cell is empty (default is an empty string '').
    contaminant (str): The string to remove from the array of elements.

    Returns:
    pd.DataFrame: The DataFrame with modified column values.
    """
    def process_cell(cell):
        # If the cell is empty, return the default value
        if pd.isna(cell) or cell == '':
            return default_value
        
        # Split the cell by the delimiter
        elements = cell.split(delimiter)
        
        # Remove empty elements and the contaminant
        elements = [elem.strip() for elem in elements if elem.strip() != '' and elem.strip() != contaminant]
        
        # Count the remaining elements
        num_elements = len(elements)
        
        # Apply the logic for the result in the cell
        if num_elements == 0:
            return '0'  # No elements left
        elif num_elements == 1:
            return '1'  # One element left
        else:
            return '2'  # Two or more elements left
    
    # Apply the function to the specified column
    data[new_column_name] = data[column_name].apply(process_cell)
    
    return data


def does_column_contain_string_in_category_list(data, column_name, new_column_name, search_list, delimiter=',', empty_value=''):
    """
    Process each cell in the specified column by splitting it by the delimiter, 
    checking if any split value matches any of the strings in the search list exactly.
    If there's a match, assign 1; if not, assign 0. If the cell is empty, return the default value.

    Args:
    data (pd.DataFrame): The DataFrame to modify.
    column_name (str): The name of the column to operate on.
    new_column_name (str): The name of the new column to save the results.
    search_list (list): A list of strings to search for in each cell.
    delimiter (str): The delimiter to split the cell by (default is comma ',').
    default_value (str): The value to use if the cell is empty (default is an empty string '').

    Returns:
    pd.DataFrame: The DataFrame with the new column added containing 1 for matches and 0 for non-matches or the default value if empty.
    """
    def process_cell(cell):
        # If the cell is empty or NaN, return the default value
        if pd.isna(cell) or cell == '':
            return empty_value
        
        # Split the cell by the delimiter
        elements = cell.split(delimiter)
        
        # Check for exact match with any of the strings in the search_list
        if any(search_string in [elem.strip() for elem in elements] for search_string in search_list):
            return 1
        else:
            return 0
    
    # Apply the function to the specified column and save results to the new column
    data[new_column_name] = data[column_name].apply(process_cell)
    
    return data

def process_column_tuples(data, start_column, columns, num_tuples, transformations=None, default_value=None, delimiter=" - "):
    """
    Processes groups of 3 columns in a DataFrame, creating new columns based on transformations.

    Args:
    - data (pd.DataFrame): The DataFrame to process.
    - start_column (int or str): The index or name of the first column of the first tuple.
    - num_tuples (int): The number of 3-column groups to process.
    - transformations (dict, optional): A dictionary for transforming values in the third column. If None, use raw values.
    - default_value (any, optional): The default value if a transformation is not found. If None, default to the original value.
    - delimiter (str): The delimiter for combining values from the first two columns (default is " - ").

    Returns:
    - pd.DataFrame: A DataFrame with the new columns added.
    """
    data_copy = data.copy()
    start_index = column_name_to_index(data_copy, start_column) if isinstance(start_column, str) else start_column

    for i in range(num_tuples):
        # Calculate indices for the current tuple
        col1_index = start_index + i * columns
        col2_index = col1_index + 1
        col3_index = col1_index + 2

        # Ensure indices are within bounds
        if col3_index >= len(data_copy.columns):
            print(f"Warning: Tuple {i+1} exceeds available columns. Stopping early.")
            break

        # Get column names
        col1_name = data_copy.columns[col1_index]
        col2_name = data_copy.columns[col2_index]
        col3_name = data_copy.columns[col3_index]

        # Process each row
        def process_row(row):
            col1_value = row.iloc[col1_index]
            col2_value = row.iloc[col2_index]
            col3_value = row.iloc[col3_index]

            # Combine col1 and col2
            if pd.notna(col1_value) and pd.notna(col2_value):
                combined_name = f"{col1_value}{delimiter}{col2_value}"
            elif pd.notna(col1_value) or pd.notna(col2_value):
                combined_name = col1_value if pd.notna(col1_value) else col2_value
                print(f"Warning: Row {row.name}, only one of {col1_name}/{col2_name} is non-empty.")
            else:
                return None  # Skip if both are empty

            # Transform col3_value
            if transformations:
                transformed_value = transformations.get(
                    str(col3_value),
                    col3_value if default_value is None else default_value
                )
            else:
                transformed_value = col3_value  # Use raw value if no transformations provided

            # Assign the value to the new column
            return combined_name, transformed_value

        # Create new column
        new_column_data = data_copy.apply(process_row, axis=1)

        # Split the processed data into names and values
        new_column_names = [item[0] if item else None for item in new_column_data]
        new_column_values = [item[1] if item else None for item in new_column_data]

        # Insert new columns dynamically
        for idx, (name, value) in enumerate(zip(new_column_names, new_column_values)):
            if name:
                data_copy.at[idx, name] = value

    return data_copy


def generate_heatmap_with_counts(data, start_column, columns_per_set, num_tuples, allow_multiple_duplicates=False, prefix_delimiter=" - ", output_file="heatmap.csv"):
    """
    Generate a heatmap matrix of unique Col1 (columns) and Col2 (rows), counting values from Col3.

    Args:
    - data (pd.DataFrame): The DataFrame to process.
    - start_column (int or str): The index or name of the first column of the first tuple.
    - columns_per_set (int): The number of columns in each set (e.g., 3 for Col1, Col2, Col3).
    - num_tuples (int): The number of column groups to process.
    - allow_multiple_duplicates (bool): If True, track all duplicate values; if False, only track the first value.
    - prefix_delimiter (str): Delimiter to join Col5 and Col2 (default " - ").
    - output_file (str): Path to save the resulting heatmap CSV.

    Returns:
    - pd.DataFrame: The heatmap matrix.
    """
    from collections import defaultdict

    # Initialize the heatmap
    heatmap = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    start_index = column_name_to_index(data, start_column) if isinstance(start_column, str) else start_column

    # Process each row
    for idx, row in data.iterrows():
        # Reset seen combinations and track duplicates for this row
        seen_combinations = {}

        # Process all column batches in this row
        for i in range(num_tuples):
            # Calculate indices for the current tuple
            col1_index = start_index + i * columns_per_set
            col2_index = col1_index + 1
            col3_index = col1_index + 2

            # If columns_per_set >= 5, include Col5 as a prefix to Col2
            col5_index = col1_index + 4 if columns_per_set >= 5 else None

            # Ensure indices are within bounds
            if col3_index >= len(data.columns):
                print(f"Warning: Tuple {i+1} exceeds available columns. Stopping early.")
                break

            # Extract and normalize values
            col1_value = row.iloc[col1_index]
            col2_value = row.iloc[col2_index]
            col3_value = row.iloc[col3_index]
            col5_value = row.iloc[col5_index] if col5_index and col5_index < len(data.columns) else None

            col1_value = col1_value if pd.notna(col1_value) and col1_value != "" else "Empty"
            col2_value = col2_value if pd.notna(col2_value) and col2_value != "" else "Empty"
            col3_value = col3_value if pd.notna(col3_value) and col3_value != "" else "Empty"
            col5_value = col5_value if pd.notna(col5_value) and col5_value != "" else ""

            # Combine Col5 and Col2 if applicable
            if col5_value:
                if col2_value != "Empty":
                    col2_value = f"{col5_value}{prefix_delimiter}{col2_value}"
                else:
                    col2_value = col5_value

            # Check for duplicates within this row
            combination_key = (col1_value, col2_value)
            if combination_key in seen_combinations:
                if not allow_multiple_duplicates:
                    # Skip updating heatmap for duplicates when not allowed
                    continue
            else:
                seen_combinations[combination_key] = col3_value

            # Update the heatmap
            heatmap[col2_value][col1_value][col3_value] += 1

    # Build the heatmap DataFrame
    unique_cols = sorted({col for row_dict in heatmap.values() for col in row_dict.keys()})
    unique_rows = sorted(heatmap.keys())
    heatmap_dict = {col1: [] for col1 in unique_cols}

    for col2_value in unique_rows:
        for col1_value in unique_cols:
            count_dict = heatmap[col2_value][col1_value]
            count_str = ", ".join(f"{k}: {v}" for k, v in count_dict.items()) if count_dict else ""
            heatmap_dict[col1_value].append(count_str)

    heatmap_df = pd.DataFrame(heatmap_dict, index=unique_rows)

    # Save to CSV
    heatmap_df.to_csv(output_file, index_label="Organism \\ Antibiotic")
    print(f"Heatmap saved to {output_file}")

    return heatmap_df

def generate_patient_specific_dataset(data, start_column, columns_per_set, num_tuples, patient_id_column, additional_fields=[], output_file="patient_dataset.csv"):
    """
    Generate a dataset where each row represents a unique Virus (Col2 value) per patient,
    mapping Antibiotic (Col1) to their Susceptibility (Col3) values.

    Args:
    - data (pd.DataFrame): The input DataFrame.
    - start_column (int or str): The index or name of the first column of the first tuple.
    - columns_per_set (int): The number of columns in each set (e.g., 5 for Virus, Antibiotic, Susceptibility, ..., AlternativeVirusName).
    - num_tuples (int): The number of column groups to process.
    - patient_id_column (str): The name of the column representing patient IDs.
    - additional_fields (list of str): List of column names to include as additional fields.
    - output_file (str): Path to save the resulting dataset.

    Returns:
    - pd.DataFrame: The transformed dataset.
    """
    start_index = column_name_to_index(data, start_column) if isinstance(start_column, str) else start_column
    patient_data = []

    # Process each row (patient)
    for idx, row in data.iterrows():
        patient_id = row[patient_id_column]
        patient_row = {field: row[field] for field in additional_fields}  # Add additional fields
        patient_row["PatientId"] = patient_id

        # Maps to store Virus→AlternativeVirusName and Virus→Antibiotic→Susceptibility mappings
        virus_map = {}
        patient_map = defaultdict(lambda: defaultdict(str))

        for i in range(num_tuples):
            # Calculate indices for the current tuple
            virus_index = start_index + i * columns_per_set + 1  # Col2 (Virus)
            antibiotic_index = start_index + i * columns_per_set  # Col1 (Antibiotic)
            susceptibility_index = start_index + i * columns_per_set + 2  # Col3 (Susceptibility)
            alternative_virus_index = start_index + i * columns_per_set + 4  # Col5 (AlternativeVirusName)

            # Ensure indices are within bounds
            if alternative_virus_index >= len(data.columns):
                print(f"Warning: Tuple {i+1} exceeds available columns. Stopping early.")
                break

            # Extract values without replacing with placeholders
            virus_value = row.iloc[virus_index] if pd.notna(row.iloc[virus_index]) and row.iloc[virus_index] != "" else None
            antibiotic_value = row.iloc[antibiotic_index] if pd.notna(row.iloc[antibiotic_index]) and row.iloc[antibiotic_index] != "" else None
            susceptibility_value = row.iloc[susceptibility_index] if pd.notna(row.iloc[susceptibility_index]) and row.iloc[susceptibility_index] != "" else None
            alternative_virus_value = row.iloc[alternative_virus_index] if pd.notna(row.iloc[alternative_virus_index]) and row.iloc[alternative_virus_index] != "" else None

            # Skip if Virus key (Col2) is empty
            if not virus_value:
                continue

            # Add alternative virus name to the map
            if virus_value not in virus_map and alternative_virus_value:
                virus_map[virus_value] = alternative_virus_value

            # Handle empty Antibiotic key (Col1)
            if not antibiotic_value:
                # Ensure the virus is added to the patient map with no antibiotic key
                if virus_value not in patient_map:
                    patient_map[virus_value] = {}
                continue  # Skip the rest of the logic for this tuple

            # Append Susceptibility value to the Virus→Antibiotic map, if it's non-empty
            if susceptibility_value:
                if antibiotic_value not in patient_map[virus_value]:
                    patient_map[virus_value][antibiotic_value] = susceptibility_value
                else:
                    patient_map[virus_value][antibiotic_value] += f", {susceptibility_value}"

        # Create rows for the new dataset
        for virus_value, antibiotic_map in patient_map.items():
            new_row = {
                "PatientId": patient_id,  # Always at the top
                "Virus": virus_value,  # Second field
                "AlternativeVirusName": virus_map.get(virus_value, ""),  # Third field
            }

            # Add patient_row fields after the first three fields
            for key, value in patient_row.items():
                new_row[key] = value

            # Populate Antibiotic values as additional columns
            for antibiotic_key, susceptibility_values in antibiotic_map.items():
                new_row[antibiotic_key] = susceptibility_values

            # Append the ordered row to patient_data
            patient_data.append(new_row)

    # Convert to DataFrame
    result_df = pd.DataFrame(patient_data)

    # Fill missing columns with empty strings
    result_df = result_df.fillna("")

    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

    return result_df


def concat_values_across_batches(data, nth_column, step_size, num_steps, output_column_name):
    """
    Concatenate values from a specific starting column (nth_column) and subsequent batches, removing duplicates and empty values.

    Args:
    - data (pd.DataFrame): The input DataFrame.
    - nth_column (int or str): The starting column for the first batch.
    - step_size (int): The number of columns to skip between batches.
    - num_steps (int): The total number of batches to process.
    - output_column_name (str): The name of the new column to store the concatenated results.

    Returns:
    - pd.DataFrame: The DataFrame with the new column added.
    """
    start_index = column_name_to_index(data, nth_column) if isinstance(nth_column, str) else nth_column
    output_values = []

    for idx, row in data.iterrows():
        value_set = set()

        for step in range(num_steps):
            column_index = start_index + step * step_size  # Calculate the column index for the current batch
            if column_index >= len(data.columns):
                print(f"Warning: Step {step + 1} exceeds available columns. Stopping early for row {idx}.")
                break

            value = row.iloc[column_index]
            if pd.notna(value) and value != "":  # Skip empty values
                value_set.add(value)

        # Join unique, non-empty values with ", "
        output_values.append(", ".join(sorted(value_set)))

    # Add the concatenated results as a new column
    data[output_column_name] = output_values
    print(f"Column '{output_column_name}' added with concatenated values.")
    return data


def extract_and_filter_raw_map(data, input_column, substrings, new_column_name):
    """
    Extract a dictionary from raw map-like data, filter by a list of substrings, and save to a new column.
    
    Args:
    - data (pd.DataFrame): The DataFrame to process.
    - input_column (str): The name of the column containing the raw map-like data.
    - substrings (list of str): The list of substrings to filter keys by (case-insensitive).
    - new_column_name (str): The name of the new column to save the filtered dictionaries.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with the new column.
    """
    substrings_lower = [s.lower() for s in substrings]

    def process_row(row):
        # Split the row into key-value tokens by ";"
        tokens = row.split(";")
        row_dict = {}
        duplicate_keys = set()

        for token in tokens:
            if "Key:" in token and "Value:" in token:
                key_value = token.split("Value:", 1)
                key = key_value[0].replace("Key:", "").strip()
                value = key_value[1].strip() if len(key_value) > 1 else ""
            elif "Key:" in token:
                key = token.replace("Key:", "").strip()
                value = ""
            else:
                continue

            # Check for duplicate keys
            if key in row_dict:
                duplicate_keys.add(key)
            row_dict[key] = value

        # Print an error if duplicate keys are found
        if duplicate_keys:
            print(f"Duplicate keys found in row:.. Duplicates: {duplicate_keys}")

        # Filter the dictionary for keys containing any of the substrings (case-insensitive)
        filtered_dict = {
            k: v for k, v in row_dict.items() if any(sub in k.lower() for sub in substrings_lower)
        }
        return filtered_dict

    # Apply the process_row function to each row
    data[new_column_name] = data[input_column].apply(process_row)
    return data

def summarize_keys_and_values_in_raw_map(data, input_column, output_file):
    """
    Summarize unique keys and their corresponding unique values in raw map-like data, 
    and save the result with flipped axis: keys as columns and unique values as rows.
    
    Args:
    - data (pd.DataFrame): The DataFrame to process.
    - input_column (str): The name of the column containing the raw map-like data.
    - output_file (str): The CSV file to save the summarized data.
    
    Returns:
    - pd.DataFrame: A summary DataFrame with keys as columns and unique values as rows.
    """
    from collections import defaultdict

    # Dictionary to store keys and sets of unique values
    key_value_map = defaultdict(set)

    def process_row(row):
        # Split the row into key-value tokens by ";"
        tokens = row.split(";")
        for token in tokens:
            if "Key:" in token and "Value:" in token:
                key_value = token.split("Value:", 1)
                key = key_value[0].replace("Key:", "").strip()
                value = key_value[1].strip() if len(key_value) > 1 else ""
            elif "Key:" in token:
                key = token.replace("Key:", "").strip()
                value = ""
            else:
                continue

            key_value_map[key].add(value)

    # Apply the process_row function to each row
    data[input_column].apply(process_row)

    # Convert key-value map to a DataFrame with keys as columns and unique values as rows
    max_values = max(len(values) for values in key_value_map.values())  # Find the max number of values for padding
    summary_data = {}

    for key, values in key_value_map.items():
        summary_data[key] = list(values) + [""] * (max_values - len(values))  # Pad with empty strings for uniform length

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    summary_df.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")

    return summary_df

def categorize_packed_cells(data, before_col, after_col, step, num_before_batches, num_after_batches, result_received, result_before, result_after):
    """
    Categorize whether a patient received packed cells before/after birth and count occurrences.
    """
    before_idx = column_name_to_index(data, before_col)
    after_idx = column_name_to_index(data, after_col)

    def process_row(row):
        count_before = 0
        count_after = 0
        #print(f"Here! NEW ROW")

        for i in range(num_before_batches):
            idx = before_idx + (i * step)
            if pd.notna(row.iloc[idx]) and str(row.iloc[idx]).strip() != "":
                #print(f"Here!: before - {row.iloc[idx]}")
                count_before += 1

        for i in range(num_after_batches):
            idx = after_idx + (i * step)
            if pd.notna(row.iloc[idx]) and str(row.iloc[idx]).strip() != "":
                #print(f"Here!: after - {row.iloc[idx]}")
                count_after += 1

        received = 1 if (count_before + count_after) > 0 else 0
        return pd.Series([received, count_before, count_after])

    data[[result_received, result_before, result_after]] = data.apply(process_row, axis=1)
    return data


def categorize_uterotonics(data, base_col, step, num_batches, result_col, cytotec_words, methergin_words):
    """
    Categorize whether a patient received uterotonics based on multiple columns.
    cytotec_words: List of words for Cytotec identification.
    methergin_words: List of words for Methergin identification.
    """
    base_idx = column_name_to_index(data, base_col)
   
    def process_row(row):
        cytotec_detected = False
        methergin_detected = False
       
        for batch in range(num_batches):
            idx = base_idx + (batch * step)
            if pd.notna(row.iloc[idx]):
                value_lower = str(row.iloc[idx]).lower()
                if any(word.lower() in value_lower for word in cytotec_words):
                    cytotec_detected = True
                if any(word.lower() in value_lower for word in methergin_words):
                    methergin_detected = True
       
        if cytotec_detected and methergin_detected:
            return 3  # Both Cytotec and Methergin received
        elif cytotec_detected:
            return 1  # Only Cytotec received
        elif methergin_detected:
            return 2  # Only Methergin received
        return 0  # No uterotonics received
   
    data[result_col] = data.apply(process_row, axis=1)
    return data

def categorize_full_dilation(data, column, result_col):
    col_idx = column_name_to_index(data, column)

    def is_full_dilation(row):
        value = row.iloc[col_idx]
        if pd.isna(value):
            return ""
        try:
            return 1 if float(value) == 10 else 0
        except:
            return ""

    data[result_col] = data.apply(is_full_dilation, axis=1)
    return data

def categorize_surgery_time(data, column, result_col):
    col_idx = column_name_to_index(data, column)
    
    def classify_time(row):
        if pd.isna(row.iloc[col_idx]) or str(row.iloc[col_idx]).strip() == "":
            return ""  # Empty if no time
        
        try:
            hour = pd.to_datetime(row.iloc[col_idx]).hour
            if 7 <= hour < 16:
                return "Day"
            elif 16 <= hour < 21:
                return "Evening"
            else:
                return "Night"
        except Exception:
            return ""  # Empty on parse failure too
    
    data[result_col] = data.apply(classify_time, axis=1)
    return data


def process_length_of_stay(data, room_col, entry_col, exit_col, step, num_batches, delivery_room_words, max_gap_minutes, result_col):
    """
    Calculate the total length of the most recent continuous stay in a delivery room.
    delivery_room_words: List of words that identify a delivery room.
    max_gap_minutes: Maximum allowed gap (in minutes) between consecutive stays for them to be considered the same sequence.
    """
    room_idx = column_name_to_index(data, room_col)
    entry_idx = column_name_to_index(data, entry_col)
    exit_idx = column_name_to_index(data, exit_col)
   
    def process_row(row):
        stays = []
       
        for batch in range(num_batches):
            idx_room = room_idx + (batch * step)
            idx_entry = entry_idx + (batch * step)
            idx_exit = exit_idx + (batch * step)
           
            if pd.notna(row.iloc[idx_room]) and any(word.lower() in str(row.iloc[idx_room]).lower() for word in delivery_room_words):
                if pd.notna(row.iloc[idx_entry]) and pd.notna(row.iloc[idx_exit]):
                    #print("entry:", row.iloc[idx_entry], row)
                    entry_time = pd.to_datetime(row.iloc[idx_entry])
                    #print("exit:", row.iloc[idx_exit])
                    exit_time = pd.to_datetime(row.iloc[idx_exit])
                    stays.append((entry_time, exit_time))
       
        # Merge continuous stays that are within max_gap_minutes apart
        stays.sort(reverse=True, key=lambda x: x[0])  # Sort stays by entry time descending
        merged_stay_time_minutes = 0.0
       
        if stays:
            current_start, current_end = stays[0]
            merged_stay_time_minutes += (current_end - current_start).total_seconds() / 60  # Keep in minutes
           
            for i in range(1, len(stays)):
                next_start, next_end = stays[i]
                if (current_start - next_end).total_seconds() / 60 <= max_gap_minutes:
                    merged_stay_time_minutes += (next_end - next_start).total_seconds() / 60  # Keep in minutes
                    current_start = next_start  # Extend the current sequence
                else:
                    break  # Stop merging if gap is too large
       
        return round(merged_stay_time_minutes / 60, 2) if merged_stay_time_minutes > 0 else None  # Convert to hours for return
   
    data[result_col] = data.apply(process_row, axis=1)
    return data

def process_length_of_fever(data, date_col, temp_col, step, num_batches, result_col):
    """
    Calculate the consecutive fever sequences (temperature above threshold).
    """
    date_idx = column_name_to_index(data, date_col)
    temp_idx = column_name_to_index(data, temp_col)

    def safe_float(val):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def fever_sequences(row):
        fever_dates = []

        for i in range(num_batches):
            idx_date = date_idx + (i * step)
            idx_temp = temp_idx + (i * step)

            temp_val = safe_float(row.iloc[idx_temp])
            date_val = row.iloc[idx_date]

            if pd.notna(date_val) and temp_val is not None and temp_val >= 38:
                try:
                    fever_dates.append(pd.to_datetime(date_val).date())
                except Exception:
                    pass  # skip invalid dates

        fever_dates = sorted(set(fever_dates))
        if not fever_dates:
            return ""

        sequences = []
        current_streak = 1
        for i in range(1, len(fever_dates)):
            if (fever_dates[i] - fever_dates[i - 1]).days == 1:
                current_streak += 1
            else:
                sequences.append(current_streak)
                current_streak = 1
        sequences.append(current_streak)

        return ", ".join(map(str, sorted(sequences, reverse=True)))

    data[result_col] = data.apply(fever_sequences, axis=1)
    return data


def process_other_cultures(data, collection_date_col, organism_col, specimen_col, step, num_batches, result_samples, result_organisms, result_organism_categories=None, organism_translation_dict=None):
    """
    Extract unique sample types and detected organisms from multiple culture test columns.
    If organism_translation_dict is provided, generate an additional column with mapped organism categories.
    """
    collection_idx = column_name_to_index(data, collection_date_col)
    organism_idx = column_name_to_index(data, organism_col)
    specimen_idx = column_name_to_index(data, specimen_col)

    def extract_culture_info(row):
        samples = set()
        organisms = set()
        for i in range(num_batches):
            date_i = collection_idx + (i * step)
            org_i = organism_idx + (i * step)
            spec_i = specimen_idx + (i * step)

            if pd.notna(row.iloc[date_i]) and str(row.iloc[date_i]).strip() != "":
                if pd.notna(row.iloc[spec_i]) and str(row.iloc[spec_i]).strip() != "":
                    samples.add(str(row.iloc[spec_i]))
                if pd.notna(row.iloc[org_i]) and str(row.iloc[org_i]).strip() != "":
                    organisms.add(str(row.iloc[org_i]))

        raw_organisms = ', '.join(organisms)
        samples_str = ', '.join(samples)

        if organism_translation_dict and result_organism_categories:
            categories = [organism_translation_dict.get(item.strip(), "") for item in organisms]
            category_str = ', '.join(filter(None, categories))
            return pd.Series([samples_str, raw_organisms, category_str])
        else:
            return pd.Series([samples_str, raw_organisms])

    if organism_translation_dict and result_organism_categories:
        data[[result_samples, result_organisms, result_organism_categories]] = data.apply(extract_culture_info, axis=1)
    else:
        data[[result_samples, result_organisms]] = data.apply(extract_culture_info, axis=1)

    return data


def imaging_guided_drainage_detected(data, static_col, repeated_col, step_size, num_steps, keywords, result_col_name):
    """
    Flags rows where imaging-guided drainage was performed based on:
    - A single static column being non-empty
    - OR any of the repeated columns (based on step) containing specific keywords
    """
    static_idx = column_name_to_index(data, static_col)
    repeated_idx = column_name_to_index(data, repeated_col)
    keywords_lower = [kw.lower() for kw in keywords]

    def check_row(row):
        # Check static column for non-empty
        if pd.notna(row.iloc[static_idx]) and str(row.iloc[static_idx]).strip() != "":
            return 1

        # Check repeated columns for keyword match
        for i in range(num_steps):
            idx = repeated_idx + i * step_size
            if idx >= len(row):
                continue
            val = str(row.iloc[idx]).lower()
            if any(kw in val for kw in keywords_lower):
                return 1

        # If nothing matched
        return 0

    data[result_col_name] = data.apply(check_row, axis=1)
    return data


def flag_infectious_indication_from_free_text(data, column_name, infectious_words, negation_prefixes, result_col, snippet_col, context_window=5):
    col_idx = column_name_to_index(data, column_name)
    infectious_words_lower = [w.lower() for w in infectious_words]
    negation_prefixes_lower = [p.lower() for p in negation_prefixes]

    def extract_match_snippet(text):
        text_lower = str(text).lower()
        words = text_lower.split()
        for i, word in enumerate(words):
            for infectious_word in infectious_words_lower:
                if infectious_word in word:
                    # Check for negation in prefix
                    prefix_slice = words[max(i - 3, 0):i]  # Up to 3 words before
                    joined_prefix = ' '.join(prefix_slice)
                    if any(neg in joined_prefix for neg in negation_prefixes_lower):
                        continue  # skip match if prefixed by negation

                    # Found a match
                    snippet_start = max(i - context_window, 0)
                    snippet = ' '.join(words[snippet_start:i + 1])
                    return (1, snippet)
        return (0, "")

    results = data.iloc[:, col_idx].apply(extract_match_snippet)
    data[result_col] = results.apply(lambda x: x[0])
    data[snippet_col] = results.apply(lambda x: x[1])
    return data


organism_dict = {
    "ACINETOBACTER SPECIES": "Other Gram Negatives",
    "ACINETOBACTER BAUMANNII-CALCOCETICUS COMPLEX": "Other Gram Negatives",
    "ACINETOBACTER BAUMANII": "Other Gram Negatives",
    "ACTINOMYCES SPECIES": "Other",
    "ACTINOTIGNUM SPECIES": "Other",
    "AEROCOCCUS SPECIES": "Anaerobes",
    "ALLOSCARDOVIA OMNICOLENS": "Anaerobes",
    "ANAEROBIC GRAM NEGATIVE ROD": "Anaerobes",
    "ANAEROBIC GRAM POSITIVE COCCUS": "Anaerobes",
    "ANAEROBIC GRAM VARIABLE ROD": "Anaerobes",
    "ASPERGILLUS FUMIGATUS": "Fungi",
    "ASPERGILLUS VERSICOLOR": "Fungi",
    "ATOPOBIUM VAGINAE": "Vaginal Flora",
    "AUREOBASIDIUM PULLULANS": "Fungi",
    "BACILLUS CEREUS": "Contaminants (CONS etc.)",
    "BACILLUS SPECIES": "Contaminants (CONS etc.)",
    "BACTEROIDES FRAGILIS GROUP": "Anaerobes",
    "BACTEROIDES FRAGILIS": "Anaerobes",
    "BACTEROIDES THETAIOTAOMICRON": "Anaerobes",
    "BIFIDOBACTERIUM SPECIES": "Anaerobes",
    "BREVIBACTERIUM SPECIES": "Contaminants (CONS etc.)",
    "BURKHOLDERIA CEPACIA": "Other Gram Negatives",
    "CANDIDA ALBICANS": "Fungi",
    "CANDIDA PARAPSILOSIS": "Fungi",
    "CAPNOCYTOPHAGA CANIMROSUS": "Other Gram Negatives",
    "CITROBACTER FREUNDII COMPLEX": "Enterobacterales",
    "CITROBACTER KOSERI": "Enterobacterales",
    "CLOSTRIDIUM PERFRINGENS": "Anaerobes",
    "CORYNEFORM BACTERIA": "Contaminants (CONS etc.)",
    "CORYNEBACTERIUM SPECIES": "Contaminants (CONS etc.)",
    "CUTIBACTERIUM (formely propionibacterium) ACNES": "Contaminants (CONS etc.)",
    "CUTIBACTERIUM ACNES": "Contaminants (CONS etc.)",
    "ENTEROBACTER AEROGENES": "Enterobacterales",
    "ENTEROBACTER CLOACAE": "Enterobacterales",
    "ENTEROBACTER CLOACAE COMPLEX": "Enterobacterales",
    "ENTEROCOCCUS FAECALIS": "Non-hemolitic Strep (viridans + enterococci)",
    "ENTEROCOCCUS SPECIES": "Non-hemolitic Strep (viridans + enterococci)",
    "ESCHERICHIA COLI": "Enterobacterales",
    "FINEGOLDIA MAGNA": "Anaerobes",
    "FUSOBACTERIUM NUCLEATUM": "Anaerobes",
    "GARDNERELLA SPEICES": "Vaginal Flora",
    "GARNERELLA SPEICES": "Vaginal Flora",
    "GARDNERELLA VAGINALIS": "Vaginal Flora",
    "GRAM NEGATIVE ROD": "Other Gram Negatives",
    "HAEMOPHILUS INFLUENZAE": "Other Gram Negatives",
    "HAEMOPHILUS PARAINFLUENZAE": "Other Gram Negatives",
    "HAEMOPHILUS SPECIES": "Other Gram Negatives",
    "H. INFLUENZAE BETA-LACTAMASE NEGATIVE": "Other Gram Negatives",
    "KLEBSIELLA AEROGENES": "Enterobacterales",
    "KLEBSIELLA PNEUMONIAE": "Enterobacterales",
    "KLEBSIELLA OXYTOCA": "Enterobacterales",
    "LACTOBACILLUS CRISPATUS": "Vaginal Flora",
    "LACTOBACILLUS SPECIES": "Vaginal Flora",
    "LISTERIA MONOCYTOGENES": "Listeria",
    "MICROCOCCUS LUTEUS": "Contaminants (CONS etc.)",
    "MICROCOCCUS SPECIES": "Contaminants (CONS etc.)",
    "MORGANELLA MORGANII": "Enterobacterales",
    "MOULD FUNGUS": "Fungi",
    "PEPTONIPHILUS ASACCHAROLYTICUS": "Anaerobes",
    "PEPTONIPHILUS SPECIES": "Anaerobes",
    "PEPTOSTREPTOCOCCUS ANAEROBIUS": "Anaerobes",
    "PEPTONIPHILUS SPECIES": "Anaerobes",
    "PEPTONIPHILUS ASACCHAROLYTICUS" :"Anaerobes",
    "PREVOTELLA BIVIA": "Anaerobes",
    "PREVOTELLA SPECIES": "Anaerobes",
    "PROPIONIBACTERIUM ACNES": "Contaminants (CONS etc.)",
    "PROPIONIBACTERIUM SPECIES": "Contaminants (CONS etc.)",
    "PROTEUS MIRABILIS": "Enterobacterales",
    "PSEUDOMONAS AERUGINOSA": "Other Gram Negatives",
    "PSEUDOMONAS SPECIES": "Other Gram Negatives",
    "PSEUDOMONAS STUTZERI": "Other Gram Negatives",
    "STAPHYLOCOCCUS AUREUS": "Staph Aureus",
    "STAPHYLOCOCCUS CAPITIS (COAG. NEG. STAPH": "Contaminants (CONS etc.)",
    "STAPHYLOCOCCUS CAPITIS": "Contaminants (CONS etc.)",
    "STAPHYLOCOCCUS COAGULASE NEGATIVE": "Contaminants (CONS etc.)",
    "STREPTOCOCCUS DYSGALACTIAE (BETA-HAEMOLYTIC)": "Haemolytic streptococci other than GBS",
    "STAPHYLOCOCCUS EPIDERMIDIS (COAG. NEG. STAPH": "Contaminants (CONS etc.)",
    "STAPHYLOCOCCUS EPIDERMIDIS": "Contaminants (CONS etc.)",
    "STAPHYLOCOCCUS HAEMOLYTICUS (COAG. NEG. STAPH": "Contaminants (CONS etc.)",
    "STAPHYLOCOCCUS HOMINIS (COAG. NEG. STAPH": "Contaminants (CONS etc.)",
    "STAPHYLOCOCCUS HOMINIS": "Contaminants (CONS etc.)",
    "STAPHYLOCOCCUS SAPROPHYTICUS": "Contaminants (CONS etc.)",
    "STAPHYLOCOCCUS WARNERI (COAG. NEG. STAPH": "Contaminants (CONS etc.)",
    "STREPTOCOCCUS AGALACTIAE (GBS)": "GBS",
    "STREPTOCOCCUS AGALACTIAE": "GBS",
    "STREPTOCOCCUS ANGINOSUS (MILLERI) GROUP": "Non-hemolitic Strep (viridans + enterococci)",
    "STREPTOCOCCUS ANGINOSUS GROUP": "Non-hemolitic Strep (viridans + enterococci)",
    "STREPTOCOCCUS ANGINOSUS": "Non-hemolitic Strep (viridans + enterococci)",
    "STREPTOCOCCUS BOVIS":"Non-hemolitic Strep (viridans + enterococci)",
    "STREPTOCOCCUS BOVIS (NOT S.GALLOLYTICUS SSP. GALLOLYTICUS)": "Non-hemolitic Strep (viridans + enterococci)",
    "STREPTOCOCCUS GALLOLYTICUS SUBSP. GALLOLYTICUS": "Non-hemolitic Strep (viridans + enterococci)",
    "STREPTOCOCCUS GALLOLYTICUS SSP. PASTEURIANUS": "Non-hemolitic Strep (viridans + enterococci)",
    "STREPTOCOCCUS GALLOLYTICUS": "Non-hemolitic Strep (viridans + enterococci)",
    "STREPTOCOCCUS MITIS GROUP": "Non-hemolitic Strep (viridans + enterococci)",
    "STREPTOCOCCUS MITIS/ORALIS": "Non-hemolitic Strep (viridans + enterococci)",
    "STREPTOCOCCUS PNEUMONIAE": "Non-hemolitic Strep (viridans + enterococci)",
    "STREPTOCOCCUS PYOGENES": "Haemolytic streptococci other than GBS",
    "STREPTOCOCCUS SALIVARIUS GROUP": "Non-hemolitic Strep (viridans + enterococci)",
    "VIRIDANS STREPTOCOCCI": "Non-hemolitic Strep (viridans + enterococci)",
    "VEILLONELLA SPECIES" : "Vaginal Flora" 
    # Anything else is set to "Other".
}



def remove_duplicates(data, columns):
    """
    Remove duplicates from a DataFrame based on a subset of columns.
    
    Args:
    data (pd.DataFrame): The input DataFrame from which to remove duplicates.
    columns (list of str): A list of Excel-style column labels to determine uniqueness.
    
    Returns:
    pd.DataFrame: A DataFrame with duplicates removed.
    """
    # Convert Excel-style labels to DataFrame column indices
    column_indices = [column_name_to_index(data, col) for col in columns]
    
    # Convert column indices to DataFrame column names
    column_names = [data.columns[idx] for idx in column_indices if idx < len(data.columns)]
    
    # Remove duplicates based on these column names
    return data.drop_duplicates(subset=column_names)

def load_data(filepath, delimiter=','):
    encodings = ['utf-8', 'windows-1255', 'utf-16', 'utf-16-le', 'utf-16-be', 'ISO-8859-1', 'ISO-8859-8', 'windows-1252']
    for encoding in encodings:
        try:
            data = pd.read_csv(filepath, encoding=encoding, delimiter=delimiter, on_bad_lines='warn', low_memory=False, keep_default_na=False, na_filter=False)
            print(f"Data loaded successfully with encoding {encoding}.")
            return data
        except UnicodeDecodeError as e:
            print(f"Error loading data with {encoding}: {e}")
        except pd.errors.ParserError as e:
            print(f"Parsing error with {encoding}: {e}")
        except Exception as e:
            print(f"An error occurred with {encoding}: {e}")
    print("Failed to load data. Please check the file path and file format.")
    return None

def save_data(data, filepath):
    try:
        data.to_csv('output.csv', encoding='utf-8', index=False)

        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data: {e}")

def split_and_save_csv(data, column_name, full_output_file, csv_file_under_38, csv_file_38_or_above, encoding='utf-8'):
    """
    Split the data into two parts based on the values in the specified column. Rows with values under 38
    go into one part, and rows with values of 38 or higher go into another part. Save both parts and the full
    dataset to separate CSV files with specified encoding and error handling.

    Args:
    data (pd.DataFrame): The DataFrame to split.
    column_name (str): The column name to split the data by.
    full_output_file (str): The file name for the full dataset (CSV format).
    csv_file_under_38 (str): The file name for rows with values under 38 (CSV format).
    csv_file_38_or_above (str): The file name for rows with values 38 or above (CSV format).
    encoding (str): The encoding format to use when saving the CSV files (default is 'utf-8').

    Returns:
    None: Saves three CSV files (full output, under 38, and 38 or above).
    """
    try:
        # Split the data into two DataFrames based on the specified column
        data_under_38 = data[data[column_name] < 38]
        data_38_or_above = data[data[column_name] >= 38]
        
        # Save the full DataFrame with specified encoding
        print(f"final length is {len(data)}")
        data.to_csv(full_output_file, index=False, encoding=encoding)
        
        # Save the two parts into separate CSV files with specified encoding
        print(f"under 38 length is {len(data_under_38)}")
        data_under_38.to_csv(csv_file_under_38, index=False, encoding=encoding)
        print(f"38 or above length is {len(data_38_or_above)}")
        data_38_or_above.to_csv(csv_file_38_or_above, index=False, encoding=encoding)
        
        print(f"Files saved successfully:\n- Full data: {full_output_file}\n- Under 38: {csv_file_under_38}\n- 38 or Above: {csv_file_38_or_above}")
        
    except Exception as e:
        print(f"An error occurred while saving the files: {e}")


def main():
    input_filepath = 'input.csv'
    output_filepath = 'output.csv'

    all_data = load_data(input_filepath)
    if all_data is None:
        print("Error, all_data is none. Tell Ariel.")
        return None
    
    de_dupe_data = remove_duplicates(all_data, ['patient id', 'parity - birth number', 'gravidity - pregnancy number'])
    print(len(all_data)-len(de_dupe_data), " rows removed in de-dupe")

    #over_threshold_data = remove_rows_below_threshold(de_dupe_data, 'birth-gestational age', 24.0)
    #print(len(de_dupe_data)-len(over_threshold_data), " rows below gestational age 24 removed")

    #data = remove_rows_if_contains(over_threshold_data, 'birth-type of labor onset', ['misoprostol', 'termination of pregnancy','IUFD'])
    #print(len(over_threshold_data)-len(data), " rows with misoprostol/termination removed")

    #data = remove_rows_above_threshold(data, 'birth-fetus count', 1)
    #print(len(over_threshold_data)-len(data), " rows with fetus count above 1 removed")

    data = de_dupe_data

    ## Cultures taken yes/no Follow by Positive yes/no
    data = containswords_result_exists(data, 'blood cultures-test type_1', ['דם'], 4, 40, 'blood_culture_taken')
    ## blood culture positive != bateremia so this column is not relevent for now
    ##data = containswords_andor_containswords_and_nonempty_result_exists(data, '', ['דם'], 'OR','', ['דם'], '', 8, 74, 'blood_culture_positive')


    ## Cultures positive organisms followed by Categories
    data = containswords_and_nonempty_result_values(data, 'blood cultures-test type_1', ['דם'], 'blood cultures-organism detected_1', 4, 40, 'blood_culture_organisms')
    data = containswords_and_nonempty_result_values(data, 'blood cultures-test type_1', ['דם'], 'blood cultures-organism detected_1', 4, 40, 'blood_culture_organisms_category', dictionary=organism_dict)
    
    data = remove_contaminant_and_count(data, 'blood_culture_organisms_category', 'Blood_culture_Type_of_growth', delimiter=',', default_value=0, contaminant='Contaminants (CONS etc.)')
    data = does_column_contain_string_in_category_list(data, 'blood_culture_organisms_category', 'Organisms_Contaminants_yes_or_no', ['Contaminants (CONS etc.)'], delimiter=',', empty_value=0)
    data = does_column_contain_string_in_category_list(data, 'blood_culture_organisms_category', 'Organisms_Non_hemolytic_Strep_yes_or_no', ['Non-hemolitic Strep (viridans + enterococci)'], delimiter=',', empty_value=0)
    data = does_column_contain_string_in_category_list(data, 'blood_culture_organisms_category', 'Organisms_Enterobacterales_yes_or_no', ['Enterobacterales'], delimiter=',', empty_value=0)
    data = does_column_contain_string_in_category_list(data, 'blood_culture_organisms_category', 'Organisms_GBS_yes_or_no', ['GBS'], delimiter=',', empty_value=0)
    data = does_column_contain_string_in_category_list(data, 'blood_culture_organisms_category', 'Organisms_Anaerobes_yes_or_no', ['Anaerobes'], delimiter=',', empty_value=0)
    data = does_column_contain_string_in_category_list(data, 'blood_culture_organisms_category', 'Organisms_Other_Gram_Negatives_yes_or_no', ['Other Gram Negatives'], delimiter=',', empty_value=0)
    data = does_column_contain_string_in_category_list(data, 'blood_culture_organisms_category', 'Organisms_Vaginal_Flora_yes_or_no', ['Vaginal Flora'], delimiter=',', empty_value=0)
    data = does_column_contain_string_in_category_list(data, 'blood_culture_organisms_category', 'Organisms_Staph_Aureus_yes_or_no', ['Staph Aureus'], delimiter=',', empty_value=0)
    data = does_column_contain_string_in_category_list(data, 'blood_culture_organisms_category', 'Organisms_Listeria_yes_or_no', ['Listeria'], delimiter=',', empty_value=0)
    data = does_column_contain_string_in_category_list(data, 'blood_culture_organisms_category', 'Organisms_Other_yes_or_no', ['Other','Uncategorized'], delimiter=',', empty_value=0)
    
    ## Antibiotics taken yes/no
    data = containswords_result_exists(data, 'antibiotics-medication_1', ['AMPICILLIN'], 3, 108, 'Antibiotics_given_Ampicillin')
    data = containswords_result_exists(data, 'antibiotics-medication_1', ['AMOXYCILLIN', 'AUGMENTIN'], 3, 108, 'Antibiotics_given_Augmentin')
    data = containswords_result_exists(data, 'antibiotics-medication_1', ['ERTAPENEM', 'MEROPENEM', 'MERONEM'], 3, 108, 'Antibiotics_given_Carbapenem')
    data = containswords_result_exists(data, 'antibiotics-medication_1', ['ROCEPHIN', 'CEFTRIAXONE'], 3, 108, 'Antibiotics_given_Ceftriaxone')
    data = containswords_result_exists(data, 'antibiotics-medication_1', ['CLINDAMYCIN', 'DALACIN'], 3, 108, 'Antibiotics_given_Clindamycin')
    data = containswords_result_exists(data, 'antibiotics-medication_1', ['GENTAMYCIN', 'GENTAMICIN'], 3, 108, 'Antibiotics_given_Gentamycin')
    data = containswords_result_exists(data, 'antibiotics-medication_1', ['FLAGYL', 'METRONIDAZOLE'], 3, 108, 'Antibiotics_given_Metronidazole')
    data = containswords_result_exists(data, 'antibiotics-medication_1', ['PENICILLIN'], 3, 108, 'Antibiotics_given_Penicillin')
    data = containswords_result_exists(data, 'antibiotics-medication_1', ['TAZOCIN', 'TAZOBACTAM+PIPERACILLIN'], 3, 108, 'Antibiotics_given_Tazocin')
    data = containswords_result_exists(data, 'antibiotics-medication_1', ['AZENIL', 'NITROFURANTOIN', 'DOXYLIN','AMIKACIN'], 3, 108, 'Antibiotics_given_Other')

    ## Antibiotics length of treatment

    ## Dictionary mapping
    #*עמודה - בשם type of labor onset
    words_dict_0 = {
        "1": ["ירידת מים", "ספונטני"],
        "2": ["ניסיון היפוך", "השראת לידה - פקיעת מים", "השראת לידה", "PG", "הבשלת צוואר", "אוגמנטציה - פיטוצין", "אוגמנטציה"],
        "3": ["הבשלת צואר - בלון"],
        "4": ["ניתוח קיסרי"],
        "5": ["אינה בלידה","אחר"],
        "6": ["Misoprostol", "Termination of pregnancy"]
    }
    update_column_with_values(data, 'type of labor onset', words_dict_0, default_value="Other")

    #*עמודה - בשם complications
    words_dict_1 = {
        "0": ["לא"],
        "1": ["כן"]
    }
    update_column_with_values(data, 'complications-value textual', words_dict_1, default_value="Other")
    
    #*עמודה - בשם amniofusion-non-numeric results
    words_dict_2 = {
        "1": ["בוצע", "בוצע עי דר הראל"]
    }
    update_column_with_values(data, 'complications-value textual', words_dict_2, default_value="Other", empty_value="0")

     #*עמודה - בשם surgery reports-surgery date-weekday
    words_dict_3 = {
        "1": ["Sunday"],
        "2": ["Monday"],
        "3": ["Tuesday"],
        "4": ["Wednesday"],
        "5": ["Thursay"],
        "6": ["Friday"],
        "7": ["Saturday"]
    }
    update_column_with_values(data, 'surgery reports-surgery date-weekday', words_dict_3, default_value="Other", empty_value="")
    
      #*עמודה - בשם surgery info-procedure
    words_dict_4 = {
        "1": ["LSCS","LOW SEGMENT CESAREAN SECTION","CESAREAN DELIVERY L.S.C.S"],
        "2": ["HIGH TRANSVERSE CESAREAN SECTION"],
        "3": ["CESAREAN HYSTERECTOMY", "CESAREAN DELIVERY AND HYSTERECTOMY"],
        "4": ["CLASSICAL"],
        "5": ["INVERTED T INCISION "]
    }
    
    update_column_with_values(data, 'surgery info-procedure', words_dict_4, default_value="Other", empty_value="")
    
    #*עמודה - בשם amniotic fluid color
    words_dict_5 = {
        "1": ["נקיים", "דמיים", "לא נצפו מים", "no value"],
        "2": ["מקוניום", "מקוניום דליל", "מקוניום סמיך"]
    }
    update_column_with_values(data, 'rom description-amniotic fluid color', words_dict_5, default_value="Other")


    #*עמודה - בשם membranes rupture type
    words_dict_6 = {
        "0": ["קרומים נפקעו עצמונית", "לא נפקעו עד ללידה", "לא נמושו קרומים", "זמן פקיעת הקרומים לא ידוע", "No value"],
        "1": ["קרומים נפקעו מכשירנית", "נפקעו לאחר בדיקה וגינלית", "נפקעו במהלך בדיקה וגינלית", "AROM"]
    }
    update_column_with_values(data, 'rom description-membranes rupture type', words_dict_6, default_value="Other")


    #*GBS בשתן וGBS בנרתיק
    words_dict_7 = {
        "0": ["שלילי"],
        "1": ["חיובי"],
        "2": ["לא נבדק", "no value", "צמיחה מעורבת"]
    }
    update_column_with_values(data, 'gbs status-gbs in urine', words_dict_7, default_value="Other")
    update_column_with_values(data, 'gbs status-gbs vagina', words_dict_7, default_value="Other")
    
    words_dict_7_5 = {
        "1": ["gbs +", "gbs carrier", "gbs carrier (v02.51)"]
    }
    update_column_with_values(data, 'gbs diagnosis-diagnosis', words_dict_7_5, default_value="Other", empty_value="2")
    
    data = custom_logic_operation(data, 'gbs diagnosis-diagnosis', 'gbs status-gbs vagina', 'GBS_vaginal_Result')

    #*עמודה - בשם Transfer to ICU
    words_dict_8 = {
        "1": ["טיפול נמרץ כללי"],
        "2": ["יחידת טראומה"]
    }
    update_column_with_values(data, 'transfer to icu-department', words_dict_8, default_value="Other", empty_value="0")

    #*עמודה - בשם readmission department
    words_dict_9 = {
        "1": ["גינקולוגיה", "יולדות א", "יולדות ב"],
        "2": ["א.א.ג ניתוחי ראש וצוואר", "אורולוגיה", "השהיה מלרד", "כירורגית ב", "כירורגית ג", "מלונית", "נוירולוגיה", "פנימית ו", "פנימית ט", "אונקולוגית", "פנימית א", "פנימית ד", "פנימית ג", "שרות שיקום מרחוק תנועה"],
    }
    #update_column_with_values(data, 'readmission-admitting department', words_dict_9, default_value="Other", empty_value="0")


    #*עמודה - בשם epidural-anesthesia type
    words_dict_10 = {
        "0": ["ללא הרדמה", "מקומית","כללית"],
        "1": ["אפידורלית", "אפידורלית+כללית", "ספינלית", "ספינלית+כללית"]
    }
    update_column_with_values(data, 'neuraxial analgesia-anesthesia type', words_dict_10, default_value="Other")

    #*עמודה AMX - בשם Surgery Indication - Main
    words_dict_11 = {
        "1": ["Breech Presentation","Brow Presentation","Cord Presentation","Face Presentation","Malpresentation","Transverse/Oblique Lie","Twins - First Non-Vertex"],
        "2": ["Non Reassuring Fetal Monitor"],
        "3": ["Arrest of dilatation","Dysfunctional Labour","Failed Induction","Failure of descent", "No progress", "Susp. CPD","Failed Vacuum Extraction","Failed Forceps extraction"],
        "4": ["MATERNAL REQUEST","Maternal Exhaustion"],
        "5": ["Prev. C.S. - Patient`s Request","Previous Uterine Scar", "S/P Myomectomy"],
        "6": ["Macrosomia"],
        "7": ["Fetal Thrombocytopenia","Marginal placenta","Multiple Pregnancy", "Other Indication", "Past Shoulder Dystocia", "Placenta Accreta", "Placenta previa", "Prolapse of Cord", "S/P Tear III/IV degree","Susp. Uterine Rupture", "Suspected Placental Abruption", "Suspected Vasa Previa", "TWINS PREGNANCY", "Genital Herpes", "Tumor Previa", "Triplet pregnancy"]
        
    }
    update_column_with_values(data, 'cs info-main indication', words_dict_11, default_value="Other")
    update_column_with_values(data, 'cs info-secondary indication', words_dict_11, default_value="Other")
    
    #עמודה בשם cs info-type of surgery
    words_dict_12 = {
        "1": ["אלקטיבי"],
        "2": ["סמי-אלקטיבי"],
        "3": ["דחוף"],
        "4": ["בהול"]
        
    }
    update_column_with_values(data, 'cs info-type of surgery', words_dict_12, default_value="Other", empty_value="")
    
     #*עמודות YN,YR,YV,YZ - בשם Procedure
     #0-No or Hysterectomy, 1-Laparotomy, 2-Laparoscoy, 3-Other
    words_dict_13 = {
        "0": ["REPAIR","HYSTERECTOMY"],
        "1": ["LAPAROTOMY","OPEN"],
        "2": ["LAPAROSCOP"],
        "3": ["WIDE","DEBRIDEMENT","BREAST","HEMATOMA","OTHER"]
      
    }
    #update_column_with_values(data, 'surgery after delivery-procedure_1', words_dict_13, default_value="Other", empty_value="0")
    #update_column_with_values(data, 'surgery after delivery-procedure_2', words_dict_13, default_value="Other", empty_value="0")

    #שימוש בהמוסטטים בניתוח
    words_dict_14 = {
        "1": ["FIBRILLAR"],
        "2": ["NU_KNIT"],
        "3": ["DETAIL_2","DiagnosisTest","Hemostasis","LyingDown","Text_1", "Sergiplo"]
    }
    update_column_with_values(data, 'hemostasis-code', words_dict_14, default_value="Other", empty_value="0")
    
    #אוגמנטציה
    words_dict_15 = {
        "1": ["MISOPROSTOL"],
        "2": ["OXYTOCIN"]
    }
    update_column_with_values(data, 'augmentation meds-medication', words_dict_15, default_value="Other", empty_value="0")
    
    #בלון/פרופס
    words_dict_16 = {
        "1": ["הכנסת בלון"],
        "2": ["הכנסת פרופס"]
    }
    update_column_with_values(data, 'balloon/propes-measurement', words_dict_16, default_value="Other", empty_value="0")
    
    ## Remove negative values from 
    #cleared = clear_negative_values(data, '')
    #print(f"{cleared} negative \"third stage length\" values removed")



    # Check if numeric values in column 'E' meet or exceed the cutoff of 1, if >= return above value, below return the below value. empty keep empty.
    #data = cutoff_number(data, 'date of death - days from delivery', 'death_at_delivery_yes_no', 1, above=0, below=1, empty_value=0)
    
    # Check if the value exists
    data = compare_values(data, 'parity - birth number', 'nulliparous_yesno', target_value=1, match_return=1, no_match_return=0)

    # Filter numbers in column 'E', removing values above 20
    #data = filter_numbers(data, 'gravidity - pregnancy number', lowerThan=0, higherThan=20)
    
    
    # Filter numbers in column 'AS', removing values below 15 and/or above 61
    data = filter_numbers(data, 'bmi-numeric result', lowerThan=15, higherThan=61)
    
    # Filter numbers in coulmn 'hospital length of stay (days)', removing values above 100
    #data = filter_numbers(data, 'hospital length of stay (days)', lowerThan=0, higherThan=100)

    # Flip the sign of numeric values in column 'AV' and remove values over 2500
    data = flip_sign(data, 'rom description-date of membranes rupture-hours from reference')
    data = filter_numbers(data, 'rom description-date of membranes rupture-hours from reference',lowerThan=0, higherThan=2500)
    
    # Check if the cell value in 'first antibiotics timing' is negative, and return 1 to a new column if it is. 
    #data = cutoff_number(data, 'first antibiotics timing calculated', 'Antibiotic_prophylaxis_yes_no', 0, above=0, below=1, empty_value='')
    # Check if the cell value in 'first antibiotics timing' is positive, and return 1 to a new column if it is. 
    #data = cutoff_number(data, 'first antibiotics timing calculated', 'Antibiotic_treatment_yes_no', 0, above=1, below=0, empty_value='')
    
    # Multiply the values in column 'ANA' by multiplier "24",if empty stays empty. creates another column for 2nd stage longer than 4 hours. then remove values above 6. 
    data = multiply_by_number(data, 'second stage length', multiplier=24)
    data = cutoff_number(data, 'second stage length', 'duration_of_2nd_stage_over_4h', 4, above=1, below=0, empty_value='')

    
    # Check if the cell value in maternal diagnosis column is empty or not, and returns 1 if its not, 0 if it is.
    data = is_empty(data, 'maternal pregestaional diabetes-maternal pregestaional diabetes-diagnosis', 'maternal_pregestational_diabetes_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'maternal gestational diabetes-maternal gestational diabetes-diagnosis', 'maternal_gestational_diabetes_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'maternal pregestational hypertension-maternal pregestational hypertension-diagnosis', 'maternal_pregestational_hypertension_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'maternal gestational hypertension-maternal gestational hypertension-diagnosis', 'maternal_gestational_hypertension_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'maternal hellp syndrome-diagnosis', 'maternal_hellp_syndrome_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'maternal pph-diagnosis', 'maternal_pph_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'blood products given-medication', 'blood_products_given_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'maternal vte_after delivery-diagnosis', 'maternal vte_after_delivery_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'maternal vte_before delivery-diagnosis', 'maternal vte_before_delivery_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'maternal infection post partum-diagnosis', 'maternal_infection_post_partum_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'hospitalization before delivery (hrp) - admission date', 'HRP_hospitalization_prepartum_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'pprom diagnosis-date of documentation', 'PPROM_yes_or_no', value_empty=0, value_not_empty=1)
    
  
    # Check if numeric values in column 'AV' meet or exceed the cutoff of __, and add results in a new column '___'
    data = cutoff_number(data, 'rom description-date of membranes rupture-hours from reference', 'duration_of_rom_over_18h', 18, above=1, below=0, empty_value='')

    # Flip the sign of numeric values in column 'BA'
    data = flip_sign(data, 'fever_max 38-43 before delivery-date of measurement-hours from reference')

    ## CT 
    
  
    # Remove values from column 'BL' if the value in column 'BI' is equal to 0.
    #data = clear_values_based_on_reference(data, 'transfers-department length of stay', reference_column_name='transfers-department', reference_value='0')
    
    # Remove specified rows, where cultures not taken
    #filtered_labor_data = remove_rows_if_contains(data, 'blood_culture_taken', ['0'])
    #print(len(data)-len(filtered_labor_data), " rows without blood culture taken removed")
    #data = filtered_labor_data
    
    #data = process_column_tuples(data, start_column="organisms susceptability-antibiotic_1", columns=6 ,num_tuples=206, transformations={"S": 1, "I": 2, "R": 3}, default_value=None)
    generate_heatmap_with_counts(data, start_column="organisms susceptability-antibiotic_1", columns_per_set=6 ,num_tuples=206, output_file="heatmap.csv")

    data = concat_values_across_batches(data, "antibiotics-medication_1", 3, 108, "concat_antibiotics_given" )  # antibiotics-medication_1 3 X 108
    #result_df = generate_patient_specific_dataset(
    #    data=data,
    #    start_column="organisms susceptability-antibiotic_1",
    #    columns_per_set=6,
    #    num_tuples=206,
    #    patient_id_column="patient id",
    #    additional_fields=["concat_antibiotics_given", "birth-gestational age", "blood_culture_organisms", "blood_culture_organisms_category"],
    #    output_file="patient_specific_dataset.csv"
    #)

    summary = summarize_keys_and_values_in_raw_map(
        data=data,
        input_column="scrub-all row data",
        output_file="key_value_summary.csv"
    )

    # Apply the function
    data = extract_and_filter_raw_map(
        data=data,
        input_column="scrub-all row data",
        substrings=["רחצה-", "חיטוי-", "Polydine"],
        new_column_name="Filtered_Keys"
    )

    data = categorize_packed_cells(data, 'packed cells before-date administered-days from reference_1', 'packed cells after-date administered-days from reference_1', step=2, 
                               num_before_batches=5, num_after_batches=3, 
                               result_received='packed_cells_received_yes_or_no', 
                               result_before='packed_cells_before_count', 
                               result_after='packed_cells_after_count')


    # Apply uterotonics categorization
    cytotec_words = ["ציטוטק", "cytotec"]
    methergin_words = ["מטרגין", "methergin"]

    data = categorize_uterotonics(data, 'uterotonics-medication_1', step=2, num_batches=2,
                              result_col='uterotonics_received',
                              cytotec_words=cytotec_words,
                              methergin_words=methergin_words)

    # Apply full dilation check
    data = categorize_full_dilation(data, 'full dilation at surgery-value numeric', 'full_dilation_at_surgery_yes_or_no')

    # Apply surgery time categorization
    data = categorize_surgery_time(data, 'surgery time-documenting date', 'surgery_time_category')

    # Apply length of stay processing
    data = process_length_of_stay(data,
                              room_col='length of stay delivery room-department_1',
                              entry_col='length of stay delivery room-room enter date_1',
                              exit_col='length of stay delivery room-room exit date_1',
                              step=5,
                              num_batches=5,
                              delivery_room_words=["חדר לידה"],
                              max_gap_minutes=20,
                              result_col='delivery_room_stay_hours')

    # Apply fever sequence processing
    data = process_length_of_fever(data,
                               date_col='count of fever over 38-date of measurement_1',
                               temp_col='count of fever over 38-numeric result_1',
                               step=3,
                               num_batches=130,
                               result_col='fever_sequences')

    # Apply culture extraction processing.
    data = process_other_cultures(data, 'other cultures-collection date-days from reference_1', 'other cultures-organism detected_1', 'other cultures-specimen material_1', 
                                      step=3, num_batches=10, result_samples='other_culture_samples_taken', 
                                      result_organisms='other_culture_organisms_detected',
                                      result_organism_categories='other_culture_categories_detected',
                                      organism_translation_dict=organism_dict)

    data = imaging_guided_drainage_detected(data,
                                                static_col="imaging_ first cti/usi-performed procedures",
                                                repeated_col="imaging_ct/cti (first 10)-performed procedures_1",
                                                step_size=4,
                                                num_steps=6,
                                                keywords=["CTI", "USI", "ניקוז"],
                                                result_col_name="Imaging_Guided_Drainage"
    )

    data = flag_infectious_indication_from_free_text(data,
                            column_name="has imaging (first ct/cti)-interpretation",
                            infectious_words=["חום", "חומים", "דלקת", "אבצס", "קולקציה", "OVT", "abscess", "fever", "inflammation", "collection", "מזוהמת", "זיהום"],
                            negation_prefixes=["ללא", "אין", "not", "no", "doesn’t", "לא נראה"],
                            result_col="Imaging_Infectious_Reason",
                            snippet_col="Infectious_Reason_Snippet"
    )


    # Remove specified columns, including single columns and ranges
    data = remove_columns(data, [
    #    'reference occurrence number',
    #    'date of birth~date of death - days from delivery',
    #    'date of first documentation - birth occurence~imaging_ first cti/usi-interpretation',
    #    'amniofusion-date of measurement-days from reference',
    #    'cs info-date of documentation~cs info-remarks',
    #    'scrub-value textual~surgery reports-documenting date',
    #    'surgery reports-complications during surgery',
    #    'full dilation at surgery-value numeric~surgery info-date of procedure',
    #    'hospitalization before delivery (hrp) - admission date~length of stay delivery room-room exit - hours from reference_5',
    #    'transfer to icu-department admission date~transfer to icu-department discharge date',
    #    'readmission-hospital admission date',
    #    'readmission-hospital discharge date',
    #    'second stage timeline-time of full dilation',
    #    'fever_max 38-43 before delivery-date of measurement',
    #    'count of fever over 38-date of measurement_1~count of fever over 38-department_130',
    #    'fever_max 38-43 after delivery-date of measurement',
    #    'pprom diagnosis-date of documentation',
    #    'pprom diagnosis-diagnosis',
    #    'maternal pregestational hypertension-maternal pregestational hypertension-diagnosis~maternal infection post partum-diagnosis',
    #    'blood cultures-organism detected_1~organisms susceptability-susceptibility value_206',
    #    'antibiotics-date administered-days from reference_1~antibiotics-medication_108',
    #    'gbs status-gbs in urine~gbs status-gbs vagina',
  
    #    'wbc max-collection date-hours from reference',
    #    'crp max-collection date-hours from reference',
    #    'surgery after delivery-department_1~surgery after delivery-department_2',
    #    'imaging-exam performed (sps)_1~imaging-performed procedures_7'
         ])

    save_data (data, 'output.csv')

if __name__ == "__main__":
    main()