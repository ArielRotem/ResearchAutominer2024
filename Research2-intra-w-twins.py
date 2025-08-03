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

def _sanitize_antibiotic_name(name):
    name = str(name).strip()
    match = re.match(r'^([A-Z+]+(?: [A-Z]+)*)', name)
    return match.group(1).strip() if match else ""

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

#def remove_rows_below_threshold(data, column_name, threshold):
    #return data[data.iloc[:, column_name_to_index(data, column_name)] >= threshold]
    
def remove_rows_below_threshold(data, column_name, threshold):
    column_index = column_name_to_index(data, column_name)
    numeric_series = pd.to_numeric(data.iloc[:, column_index], errors='coerce')  # Convert to float, invalid strings become NaN
    return data[numeric_series >= threshold]
    
#def remove_rows_above_threshold(data, column_name, threshold):
    #return data[data.iloc[:, column_name_to_index(data, column_name)] <= threshold]

def remove_rows_above_threshold(data, column_name, threshold):
    column_index = column_name_to_index(data, column_name)
    numeric_series = pd.to_numeric(data.iloc[:, column_index], errors='coerce')
    return data[numeric_series <= threshold]

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
    
def update_column_with_values_batch(inputdata, column_name, words_dict, default_value="0", empty_value=None, batch=1):
    data = inputdata.copy()
    for i in range(1, batch + 1):
        update_column_with_values(data, f"{column_name}_{i}", words_dict, default_value, empty_value)
    return data
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

def filter_numbers(data, column_name, lowerThan=None, higherThan=None):
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

def cutoff_range_numeric(data, column_name, new_column_name, lower_bound, upper_bound, in_range=1, out_of_range=0, empty_value=None):
    """
    Creates a new column indicating whether values fall within a numeric range.

    Args:
    data (pd.DataFrame): The DataFrame to modify.
    column_name (str): Column name to check.
    new_column_name (str): Name of the new column to create.
    lower_bound (float): Lower bound of the acceptable range (inclusive).
    upper_bound (float): Upper bound of the acceptable range (inclusive).
    in_range (int): Value to assign if the cell value is within the range.
    out_of_range (int): Value to assign if the cell value is outside the range.
    empty_value (any): Value to assign if the original cell is empty or non-numeric.

    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    column_index = column_name_to_index(data, column_name)

    def check_range(x):
        try:
            x = float(x)
            return in_range if lower_bound <= x <= upper_bound else out_of_range
        except ValueError:
            return empty_value

    data[new_column_name] = data.iloc[:, column_index].apply(check_range)
    return data

def compare_values(data, column_name, new_column_name, target_value, match_return, no_match_return):
    """
    Creates a new column in a DataFrame to indicate whether cell values in a specified column 
    match a given value, considering type coercion to handle both numeric and string comparisons.
    Returns 'match_return' if they match, 'no_match_return' otherwise.

    Args:
    data (pd.DataFrame): The DataFrame to modify.
    column_name (str): Name of the column to check.
    new_column_name (str): Name of the new column to create.
    target_value (any): Value to compare against the cell values.
    match_return (any): Value to return if the cell matches target_value.
    no_match_return (any): Value to return if the cell does not match target_value.
    
    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    column_index = column_name_to_index(data, column_name)
    
    def check_match(x):
        if pd.isna(x) or str(x).strip() == "":
            return x  # Leave empty cells unchanged                                
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
        #print(f"find me:{filtered_values}")
        filtered_values = list(set(filtered_values))
        #print(f"find me 2:{filtered_values}")

        # 
        # Remove '0'-like values only if other values exist
        if len(filtered_values) > 1:
            filtered_values = [v for v in filtered_values if v not in {'0', '0.0', 0}]
        #print(f"find me 3:{filtered_values}")


        
        # Join values with the specified delimiter
        return str(delimiter.join(filtered_values))

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
        if patient_id != "416312D000000000A1641D1357E4E3A61A6B32BC96F2B4C239044DF1832D6C1190AB6BA37DA0721B":
            continue
        
        print("HERE !!!!!!!!!!!!!!!  ", patient_id)
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
                #print("1")
                continue

            # Add alternative virus name to the map
            if virus_value not in virus_map and alternative_virus_value:
                virus_map[virus_value] = alternative_virus_value
                print("2")

            # Handle empty Antibiotic key (Col1)
            if not antibiotic_value:
                # Ensure the virus is added to the patient map with no antibiotic key
                print("3")
                if virus_value not in patient_map:
                    patient_map[virus_value] = {}
                    print("4")
                continue  # Skip the rest of the logic for this tuple

            # Append Susceptibility value to the Virus→Antibiotic map, if it's non-empty
            if susceptibility_value:
                print("5")
                if antibiotic_value not in patient_map[virus_value]:
                    print("6")
                    patient_map[virus_value][antibiotic_value] = susceptibility_value
                else:
                    print("6.2")
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

def split_gestational_age(data, column_name='gestational age', week_col='gestational_week', day_col='gestational_day'):
    """
    Splits a gestational age column in dot-decimal format (e.g., 38.5 means 38 weeks + 5 days)
    into two new columns: gestational_week and gestational_day.

    Args:
    data (pd.DataFrame): The input DataFrame.
    column_name (str): Name of the column with gestational age in week.day format.
    week_col (str): Name for the new weeks column.
    day_col (str): Name for the new days column.

    Returns:
    pd.DataFrame: The modified DataFrame with two new columns.
    """
    column_index = column_name_to_index(data, column_name)

    def extract_week(x):
        try:
            return int(float(x))
        except:
            return ""

    def extract_day(x):
        try:
            # Get the digits after the decimal point, then round just in case of 38.1999
            return int(round((float(x) - int(float(x))) * 10))
        except:
            return ""

    data[week_col] = data.iloc[:, column_index].apply(extract_week)
    data[day_col] = data.iloc[:, column_index].apply(extract_day)

    print(f"Added columns: {week_col}, {day_col}")
    return data

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

def add_baby_info_to_mothers(
    mothers_df,
    babies_file_path,
    mother_baby_id_columns,
    babies_id_column="patient_id",
    babies_prefix="baby_"
):
    """
    For each mother in mothers_df, looks up baby info from a babies file and adds columns
    for each baby's details, prefixed by baby index.

    mothers_df: DataFrame with mother info and baby ID columns
    babies_file_path: path to babies file (CSV)
    mother_baby_id_columns: list of 3 columns in mothers_df that contain baby IDs
    babies_id_column: column name in babies file containing the unique baby ID
    babies_prefix: prefix to add to inserted columns (default "baby_")

    Returns: mothers_df with new baby info columns added
    """
    babies_df = load_data(babies_file_path)
    babies_df = babies_df.astype(str)
    babies_df = babies_df.set_index(babies_id_column)
    babies_columns = [col for col in babies_df.columns if col != babies_id_column]

    matched_babies = 0
    matched_mothers = 0

    # Will mark True for each mother that had at least one match
    mother_matched_any = []

    for i, colname in enumerate(mother_baby_id_columns, 1):
        id_col = colname
        prefix = f"{babies_prefix}{i}_"
        for baby_col in babies_columns:
            new_col = f"{prefix}{baby_col}"
            mothers_df[new_col] = ""

    def add_baby_data_for_mother(row):
        matched_this_mother = False
        baby_results = {}
        for i, colname in enumerate(mother_baby_id_columns, 1):
            prefix = f"{babies_prefix}{i}_"
            baby_id = str(row[colname]).strip()
            if not baby_id or baby_id.lower() in ("nan", "none"):
                for baby_col in babies_columns:
                    baby_results[f"{prefix}{baby_col}"] = ""
                continue
            if baby_id in babies_df.index:
                nonlocal matched_babies
                matched_babies += 1
                matched_this_mother = True
                for baby_col in babies_columns:
                    baby_results[f"{prefix}{baby_col}"] = babies_df.at[baby_id, baby_col]
            else:
                for baby_col in babies_columns:
                    baby_results[f"{prefix}{baby_col}"] = ""
        mother_matched_any.append(matched_this_mother)
        return pd.Series(baby_results)

    baby_data = mothers_df.apply(add_baby_data_for_mother, axis=1)
    mothers_df.update(baby_data)
    matched_mothers = sum(mother_matched_any)

    print(f"Matched {matched_babies} babies to {matched_mothers} mothers (patients) out of {len(mothers_df)}.")
    return mothers_df

def add_days_between_flag(data, birth_col, death_col, result_col):
    """
    Adds a column to the dataframe: 1 if days between birth and death are 0-28 inclusive, else 0.
    Handles date strings and numeric days.
    """
    def parse_date(val):
        # Try to parse as date, fallback to float if possible
        try:
            return pd.to_datetime(val, dayfirst=True)
        except Exception:
            try:
                return float(val)
            except Exception:
                return pd.NaT  # Not a Time / missing
    
    def check_days(row):
        b = row[birth_col]
        d = row[death_col]
        b_parsed = parse_date(b)
        d_parsed = parse_date(d)
        if pd.isna(b_parsed) or pd.isna(d_parsed):
            return 0
        # If both parsed as Timestamps, subtract
        if isinstance(b_parsed, pd.Timestamp) and isinstance(d_parsed, pd.Timestamp):
            delta = (d_parsed - b_parsed).days
        else:
            try:
                delta = float(d_parsed) - float(b_parsed)
            except Exception:
                return 0
        return 1 if (delta >= 0 and delta <= 28) else 0

    data[result_col] = data.apply(check_days, axis=1)
    return data

def flag_antibiotic_within_timeframe_idx(inputdata, event_date_col, abx_med_col, abx_date_col, num_abx, step_size, timeframe, output_col, antibiotics_to_include=None, alternative_event_date_col=""):
    data = inputdata.copy()
    abx_med_start = data.columns.get_loc(abx_med_col)
    abx_date_start = data.columns.get_loc(abx_date_col)

    if antibiotics_to_include:
        abx_set = {_sanitize_antibiotic_name(abx) for abx in antibiotics_to_include}
        def abx_match(name):
            return _sanitize_antibiotic_name(name) in abx_set
    else:
        def abx_match(name):
            return True

    def check_time_in_window(row, event_time, ref_time):
        for i in range(num_abx):
            abx = row.iloc[abx_med_start + i * step_size]
            abx_date = row.iloc[abx_date_start + i * step_size]
            if pd.isna(abx) or str(abx).strip() == "" or not abx_match(abx):
                continue
            if pd.isna(abx_date) or str(abx_date).strip() == "":
                continue
            try:
                abx_date_val = pd.to_datetime(abx_date, dayfirst=True)
            except Exception:
                continue
            window_start = min(event_time, ref_time)
            window_end = max(event_time, ref_time)
            if window_start <= abx_date_val <= window_end:
                return True
        return False

    if isinstance(timeframe, str):
        def has_abx(row):
            event_time = row.get(event_date_col, None)
            if (pd.isna(event_time) or event_time == "") and alternative_event_date_col != "":
                event_time = row.get(alternative_event_date_col, None)

            ref_time = row.get(timeframe, None)
            if pd.isna(event_time) or pd.isna(ref_time) or event_time == "" or ref_time == "":
                return "0"
            try:
                event_time = pd.to_datetime(event_time, dayfirst=True)
                ref_time = pd.to_datetime(ref_time, dayfirst=True)
            except Exception:
                return "0"
            return "1" if check_time_in_window(row, event_time, ref_time) else "0"
    else:
        def has_abx(row):
            event_time = row.get(event_date_col, None)
            if pd.isna(event_time) or event_time == "":
                return "0"
            try:
                event_time = pd.to_datetime(event_time, dayfirst=True)
            except Exception:
                return "0"
            # Add or subtract the hour offset using pd.Timedelta
            ref_time = event_time + pd.Timedelta(hours=timeframe)
            return "1" if check_time_in_window(row, event_time, ref_time) else "0"

    data[output_col] = data.apply(has_abx, axis=1)
    return data


def replace_column_spaces(data, replacement="_"):
    """
    Replace all spaces in DataFrame column names with the given replacement character/string.
    Operates in-place and preserves column order.
    """
    data.columns = [col.replace(" ", replacement) for col in data.columns]
    return data

def time_to_treatment_after_event(
    inputdata,
    event_date_col,
    abx_med_col,
    abx_date_col,
    num_abx,
    step_size,
    result_col,
    antibiotics_to_include=None
):
    """
    For each row, finds the time in hours to the closest antibiotic (optionally filtered) given *after* the event date,
    and records the normalized name of that antibiotic.
    """
    data = inputdata.copy()
    abx_med_start = data.columns.get_loc(abx_med_col)
    abx_date_start = data.columns.get_loc(abx_date_col)
    event_col_idx = data.columns.get_loc(event_date_col)

    if antibiotics_to_include:
        abx_set = {_sanitize_antibiotic_name(a) for a in antibiotics_to_include}
        def abx_match(name):
            return _sanitize_antibiotic_name(name) in abx_set
    else:
        def abx_match(name):
            return True

    def find_time_and_abx(row):
        try:
            event_time = pd.to_datetime(row.iloc[event_col_idx], dayfirst=True)
        except Exception:
            return ("", "")
        candidates = []
        for i in range(num_abx):
            abx = row.iloc[abx_med_start + i * step_size]
            abx_time = row.iloc[abx_date_start + i * step_size]
            if pd.isna(abx) or pd.isna(abx_time) or str(abx).strip() == "" or str(abx_time).strip() == "":
                continue
            if not abx_match(abx):
                continue
            try:
                abx_time_val = pd.to_datetime(abx_time, dayfirst=True)
            except Exception:
                continue
            delta_hours = (abx_time_val - event_time).total_seconds() / 3600
            if delta_hours >= 0:
                candidates.append((delta_hours, _sanitize_antibiotic_name(abx)))
        if not candidates:
            return ("", "")
        best = sorted(candidates)[0]
        return best  # (hours, abx name)

    # Unpack tuple into two columns
    data[[result_col, result_col + "_abx"]] = data.apply(
        lambda row: pd.Series(find_time_and_abx(row)), axis=1
    )
    return data
    
def calculate_days_between_dates(data, start_col, end_col, result_col):
    """
    Calculates the number of days between two date columns.
    If either date is missing or invalid, the result is NaN.
    """
    data[result_col] = pd.to_datetime(data[end_col], errors='coerce', dayfirst=True) - pd.to_datetime(data[start_col], errors='coerce', dayfirst=True)
    data[result_col] = data[result_col].dt.days
    return data
    
def calculate_exact_days_with_fallback(data, start_col, end_col, alternate_end_col, result_col):
    """
    Calculates the exact number of days (as a float) between two datetime columns.
    If the result is negative, it recalculates using an alternate end column.
    """
    # Convert all datetime columns using dayfirst=True
    start_dates = pd.to_datetime(data[start_col], errors='coerce', dayfirst=True)
    end_dates = pd.to_datetime(data[end_col], errors='coerce', dayfirst=True)
    alt_end_dates = pd.to_datetime(data[alternate_end_col], errors='coerce', dayfirst=True)

    # Calculate timedelta in days (from seconds)
    delta = (end_dates - start_dates).dt.total_seconds() / (60 * 60 * 24)

    # Fallback: use alternate end date if result is negative
    fallback_delta = (alt_end_dates - start_dates).dt.total_seconds() / (60 * 60 * 24)
    final_delta = delta.where(delta >= 0, fallback_delta)

    # Store result in new column
    data[result_col] = final_delta
    return data
    

def split_twin_rows(data,
                    b1b1_first_col, b1b1_num_cols, b1b1_rm_prefix,
                    b1b2_first_col, b1b2_num_cols, b1b2_rm_prefix,
                    b2b1_first_col, b2b1_num_cols, b2b1_rm_prefix,
                    b2b2_first_col, b2b2_num_cols, b2b2_rm_prefix,
                    b3b1_first_col=None, b3b1_num_cols=None, b3b1_rm_prefix=None,
                    b3b2_first_col=None, b3b2_num_cols=None, b3b2_rm_prefix=None):

    def copy_columns(row, first_col, num_cols, rm_prefix, target_row, idx=0):
        cols = df.columns[df.columns.get_loc(first_col): df.columns.get_loc(first_col) + num_cols]
        generic_cols = [re.sub(f'^{re.escape(rm_prefix)}|{re.escape(rm_prefix)}$', '', col) for col in cols]
        for col_from, col_to in zip(cols, generic_cols):
            target_row[col_to] = row[col_from]
        target_row["twin_index"] = idx

    df = data.copy()
    new_rows = []
    babies_split = 0

    for idx, row in df.iterrows():
        original_row = row.copy()
        # Edit the original row in place for baby 1
        copy_columns(row, b1b1_first_col, b1b1_num_cols, b1b1_rm_prefix, row, 1)
        copy_columns(row, b1b2_first_col, b1b2_num_cols, b1b2_rm_prefix, row, 1)
        row["has_twin"] = 0
        new_rows.append(row)

        # Prepare baby 2 column slices
        start_b2b1 = df.columns.get_loc(b2b1_first_col)
        cols_b2b1 = df.columns[start_b2b1:start_b2b1 + b2b1_num_cols]
        start_b2b2 = df.columns.get_loc(b2b2_first_col)
        cols_b2b2 = df.columns[start_b2b2:start_b2b2 + b2b2_num_cols]

        # Add a new row if baby 2 exists
        if original_row[cols_b2b1].notnull().any() or original_row[cols_b2b2].notnull().any():
            row_baby2 = original_row.copy()
            copy_columns(original_row, b2b1_first_col, b2b1_num_cols, b2b1_rm_prefix, row_baby2, 2)
            copy_columns(original_row, b2b2_first_col, b2b2_num_cols, b2b2_rm_prefix, row_baby2, 2)
            row["has_twin"] = 1
            row_baby2["has_twin"] = 1
            new_rows.append(row_baby2)
            babies_split += 1

        # Add a new row if baby 3 exists
        if b3b1_first_col and b3b2_first_col:
            start_b3b1 = df.columns.get_loc(b3b1_first_col)
            cols_b3b1 = df.columns[start_b3b1:start_b3b1 + b3b1_num_cols]
            start_b3b2 = df.columns.get_loc(b3b2_first_col)
            cols_b3b2 = df.columns[start_b3b2:start_b3b2 + b3b2_num_cols]

            if original_row[cols_b3b1].notnull().any() or original_row[cols_b3b2].notnull().any():
                row_baby3 = original_row.copy()
                copy_columns(original_row, b3b1_first_col, b3b1_num_cols, b3b1_rm_prefix, row_baby3, 3)
                copy_columns(original_row, b3b2_first_col, b3b2_num_cols, b3b2_rm_prefix, row_baby3, 3)
                row["has_twin"] = 1
                row_baby3["has_twin"] = 1
                new_rows.append(row_baby3)
                babies_split += 1

    new_df = pd.DataFrame(new_rows)

    print(f"{babies_split} babies split from their main row")

    return new_df


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

def save_data(data, filepath='output.csv'):
    try:
        data.to_csv(filepath, encoding='utf-8', index=False)

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
        
        
   
def filter_newborn_data_by_existing_patients(newborn_filepath, current_data, output_filepath="filtered_newborn_data.csv"):
    """
    Loads a newborn data file and retains only rows where 'patient ID - mother' 
    exists in the 'patient id' column of the current dataset.

    Args:
    newborn_filepath (str): Path to the newborn data CSV file.
    current_data (pd.DataFrame): The DataFrame containing the original patient list.
    output_filepath (str): Optional. Where to save the filtered newborn data.

    Returns:
    pd.DataFrame: Filtered newborn data.
    """
    # Load newborn data
    newborn_data = load_data(newborn_filepath)
    if newborn_data is None:
        print("Failed to load newborn data.")
        return None

    original_count = len(newborn_data)

    # Filter the data
    filtered_data = newborn_data[newborn_data['cohort reference event-patient id (mother)'].isin(current_data['patient id'])]

    final_count = len(filtered_data)
    deleted_count = original_count - final_count

    print(f"Original rows: {original_count}")
    print(f"Deleted rows: {deleted_count}")
    print(f"Final rows: {final_count}")

    # Save the filtered result
    filtered_data.to_csv(output_filepath, index=False, encoding='utf-8')
    print(f"Filtered newborn data saved to {output_filepath}")

    return filtered_data


def main():
    input_filepath = 'input.csv'
    output_filepath = 'output2.csv'

    all_data = load_data(input_filepath)
    if all_data is None:
        print("Error, all_data is none. Tell Ariel.")
        return None
    
    de_dupe_data = remove_duplicates(all_data, ['patient id', 'birth-birth number', 'birth-pregnancy number', 'obstetric formula-number of births (p)', 'obstetric formula-number of pregnancies (g)'])
    print(len(all_data)-len(de_dupe_data), " rows removed in de-dupe")

    over_38_data = remove_rows_below_threshold(de_dupe_data, 'fever temperature numeric_max 37.5-43-numeric result', 38)
    print(len(de_dupe_data)-len(over_38_data), " rows below Temp 38 removed")
    
    unique_patients = remove_duplicates(over_38_data, ['patient id'])
    
    print(f"Study starting with {len(over_38_data)} deliveries with intrapartum fever - {len(unique_patients)} patients")
    
    over_threshold_data = remove_rows_below_threshold(over_38_data, 'birth-gestational age', 24.0)
    print(len(over_38_data)-len(over_threshold_data), " rows below gestational age 24 removed")

    no_termination_data = remove_rows_if_contains(over_threshold_data, 'birth-type of labor onset', ['misoprostol', 'termination of pregnancy','IUFD'])
    print(len(over_threshold_data)-len(no_termination_data), " rows with misoprostol/termination removed")

    #data = remove_rows_above_threshold(data, 'birth-fetus count', 1)
    #print(len(over_threshold_data)-len(data), " rows with fetus count above 1 removed")
    below_threshold_data = remove_rows_below_threshold(no_termination_data, 'birth-fetus count', 0)
    print(len(no_termination_data)-len(below_threshold_data), " rows with fetus count below 0 removed")
  
  ## Cultures taken yes/no Follow by Positive yes/no
    data = containswords_andor_containswords_result_exists(below_threshold_data, 'cultures-test type_1', ['דם'], 'OR','cultures-specimen material_1', ['דם'], 8, 61, 'blood_culture_taken')
    ## blood culture positive != bateremia so this column is not relevent for now
    ##data = containswords_andor_containswords_and_nonempty_result_exists(data, '', ['דם'], 'OR','', ['דם'], '', 8, 74, 'blood_culture_positive')
    
    #data = containswords_andor_containswords_result_exists(data, '', ['שתן'], 'OR','', ['שתן'], 8, 74, 'urine_culture_taken')
    #data = containswords_andor_containswords_and_nonempty_result_exists(data, '', ['שתן'], 'OR','', ['שתן'], '', 8, 74, 'urine_culture_positive')
    
    #data = containswords_andor_containswords_result_exists(data, '', ['שליה', 'שיליה'], 'OR','', ['שליה', 'שיליה'], 8, 74, 'placenta_culture_taken')
    #data = containswords_andor_containswords_and_nonempty_result_exists(data, '', [], '','', ['שליה', 'שיליה'], '', 8, 74, 'placenta_culture_positive')
    
    #data = containswords_andor_containswords_result_exists(data, '', ['מי שפיר'], 'OR','', ['מי שפיר'], 8, 74, 'amniotic_fluid_culture_taken')
    #data = containswords_andor_containswords_and_nonempty_result_exists(data, '', [], '','', ['מי שפיר'], '', 8, 74,  'amniotic_fluid_culture_positive')
    
    #data = containswords_andor_containswords_result_exists(data, '', ['לדן', 'צואר'], 'OR','', ['לדן', 'צואר'], 8, 74, 'vaginal_culture_taken')
    #data = containswords_andor_containswords_and_nonempty_result_exists(data, '', [], '','', ['לדן', 'צואר'], '', 8, 74, 'vaginal_culture_positive')



    ## Cultures positive organisms followed by Categories
    data = containswords_andor_containswords_and_nonempty_result_values(data, 'cultures-test type_1', ['דם'], 'OR','cultures-specimen material_1', ['דם'], 'cultures-organism detected_1', 8, 61, 'blood_culture_organisms')
    data = containswords_andor_containswords_and_nonempty_result_values(data, 'cultures-test type_1', ['דם'], 'OR','cultures-specimen material_1', ['דם'], 'cultures-organism detected_1', 8, 61, 'blood_culture_organisms_category', dictionary=organism_dict)
    
    #data = containswords_andor_containswords_and_nonempty_result_values(data, '', ['שתן'], 'OR','', ['שתן'], '', 8, 74, 'urine_culture_organisms')
    #data = containswords_andor_containswords_and_nonempty_result_values(data, '', ['שתן'], 'OR','', ['שתן'], '', 8, 74, 'urine_culture_organisms_category', dictionary=organism_dict)
    
    #data = containswords_andor_containswords_and_nonempty_result_values(data, '', [], '','', ['שליה', 'שיליה'], '', 8, 74, 'placenta_culture_organisms')
    #data = containswords_andor_containswords_and_nonempty_result_values(data, '', [], '','', ['שליה', 'שיליה'], '', 8, 74, 'placenta_culture_organisms_category', dictionary=organism_dict)
    
    #data = containswords_andor_containswords_and_nonempty_result_values(data, '', [], '','', ['מי שפיר'], '', 8, 74,  'amniotic_fluid_culture_organisms')
    #data = containswords_andor_containswords_and_nonempty_result_values(data, '', [], '','', ['מי שפיר'], '', 8, 74,  'amniotic_fluid_culture_organisms_category', dictionary=organism_dict)
    
    #data = containswords_andor_containswords_and_nonempty_result_values(data, '', [], '','', ['לדן', 'צואר'], '', 8, 74,  'vaginal_culture_organisms')
    #data = containswords_andor_containswords_and_nonempty_result_values(data, '', [], '','', ['לדן', 'צואר'], '', 8, 74, 'vaginal_culture_organisms_category', dictionary=organism_dict)
    
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

    ## Dictionary mapping
    #*עמודה J - בשם type of labor onset
    words_dict_0 = {
        "1": ["ירידת מים", "ספונטני"],
        "2": ["ניסיון היפוך", "השראת לידה - פקיעת מים", "השראת לידה", "PG", "הבשלת צוואר", "אוגמנטציה - פיטוצין", "אוגמנטציה"],
        "3": ["הבשלת צואר - בלון"],
        "4": ["ניתוח קיסרי"],
        "5": ["אינה בלידה","אחר"]
    }
    update_column_with_values(data, 'birth-type of labor onset', words_dict_0, default_value="Other")

    #*עמודה L - בשם birth site  
    words_dict_1 = {
        "1": ["אוטו", "אמבולנס"],
        "0": ["חדר לידה", "חדר ניתוח", "מחלקה אחרת", "מרכז לידה"]
    }
    update_column_with_values(data, 'birth-birth site', words_dict_1, default_value="Other")


    #*עמודה Q - בשם pregnancy type
    words_dict_2 = {
        "1": ["IUI", "IVF", "IVF-PGD", "איקקלומין", "גונדוטרופינים", "גונדוטרופינים + IUI", "טיפול הורמונלי - גונדוטרופינים", "טיפול הורמונלי - כלומיפן", "כלומיפן + IUI", "לטרזול", "כוריגון", "אחר"],
        "0": ["עצמוני"]
    }
    update_column_with_values(data, 'pregnancy_conceive-pregnancy type', words_dict_2, default_value="Other")


    #*עמודה AC - בשם newborn died
    words_dict_3 = {
        "0": ["No"],
        "1": ["Yes"]
    }
    update_column_with_values(data, 'newborn sheet-died at pregnancy/birth_1', words_dict_3, default_value="Other")
    update_column_with_values(data, 'newborn sheet-died at pregnancy/birth_2', words_dict_3, default_value="Other")
    update_column_with_values(data, 'newborn sheet-died at pregnancy/birth_3', words_dict_3, default_value="Other")

    #*עמודה AD - בשם newborn gender
    words_dict_4 = {
        "1": ["Female", "נקבה"],
        "2": ["Male", "זכר"]
    }
    update_column_with_values(data, 'newborn sheet-gender_1', words_dict_4, default_value="Other")
    update_column_with_values(data, 'newborn sheet-gender_2', words_dict_4, default_value="Other")
    update_column_with_values(data, 'newborn sheet-gender_3', words_dict_4, default_value="Other")
   

    #*עמודה AG - בשם newborn sent to intensive care
    update_column_with_values(data, 'newborn sheet-sent to intensive care_1', words_dict_3, default_value="Other")
    update_column_with_values(data, 'newborn sheet-sent to intensive care_2', words_dict_3, default_value="Other")
    update_column_with_values(data, 'newborn sheet-sent to intensive care_3', words_dict_3, default_value="Other")
   
    
     #*עמודה AH - בשם Mode of delivery
    words_dict_12 = {
        "0": ["רגילה","Assisted breech delivery","Spontabeous breech delivery","Total breech extraction", "עכוז"],
        "1": ["וואקום","מלקחיים"],
        "2": ["קיסרי"]
      
    }
    update_column_with_values(data, 'newborn sheet-delivery type_1', words_dict_12, default_value="Other")
    update_column_with_values(data, 'newborn sheet-delivery type_2', words_dict_12, default_value="Other")
    update_column_with_values(data, 'newborn sheet-delivery type_3', words_dict_12, default_value="Other")
    
    #*עמודה AU - בשם amniotic fluid color
    words_dict_5 = {
        "0": ["נקיים", "דמיים", "לא נצפו מים", "no value"],
        "1": ["מקוניום", "מקוניום דליל", "מקוניום סמיך"]
    }
    update_column_with_values(data, 'rom description-amniotic fluid color', words_dict_5, default_value="Other")


    #*עמודה AW - בשם membranes rupture type
    words_dict_6 = {
        "0": ["קרומים נפקעו עצמונית", "לא נפקעו עד ללידה", "לא נמושו קרומים", "זמן פקיעת הקרומים לא ידוע", "No value"],
        "1": ["קרומים נפקעו מכשירנית", "נפקעו לאחר בדיקה וגינלית", "נפקעו במהלך בדיקה וגינלית", "AROM"]
    }
    update_column_with_values(data, 'rom description-membranes rupture type', words_dict_6, default_value="Other")


    #*עמודות AY,AZ - עם השמות GBS בשתן וGBS בנרתיק בהתאמה
    words_dict_7 = {
        "0": ["שלילי"],
        "1": ["חיובי"],
        "2": ["לא נבדק", "no value", "צמיחה מעורבת"]
    }
    update_column_with_values(data, 'gbs status-gbs in urine', words_dict_7, default_value="Other")
    update_column_with_values(data, 'gbs status-gbs vagina', words_dict_7, default_value="Other")
    
    data = custom_logic_operation(data, 'gbs status-gbs in urine', 'gbs status-gbs vagina', 'GBS_Result')

    #*עמודה BI - בשם Transfer department
    words_dict_8 = {
        "0": ["גינקולוגיה", "השהייה במיון גניקולוגי", "התאוששות מיילדות", "יולדות א", "יולדות ב", "מלונית"],
        "1": ["טיפול נמרץ כללי"],
        "2": ["טיפול נמרץ קורונה", "פנימית ה", "פנימית א", "פנימית ו", "כירורגית ב","טיפול נמרץ לב", "כירורגיה כללית", "מחלקה נוירוכירורגיה", "כירורגית חזה-כלי דם", "יחידת טראומה", "טיפול נמרץ ניתוחי לב", "מחלקת קרדיולוגיה"]
    }
    update_column_with_values(data, 'transfers-department', words_dict_8, default_value="Other", empty_value="0")

    #*עמודה BO - בשם readmission department
    words_dict_9 = {
        "0": ["א.א.ג ניתוחי ראש וצוואר", "אורולוגיה", "השהיה מלרד", "כירורגית ב", "כירורגית ג", "מלונית", "נוירולוגיה", "פנימית ו", "פנימית ט", "פנימית ד", "כירורגיה כללית", "אונקולוגית", "שבץ ומחלות נוירווסקולריות", "מחלקת קרדיולוגיה" ,"עור ומין", "פנימית ה", "מחלקה נוירוכירורגיה"],
        "1": ["גינקולוגיה", "יולדות א", "יולדות ב"]
    }
    update_column_with_values(data, 'readmission-admitting department', words_dict_9, default_value="Other", empty_value="0")


    #*עמודה AMW - בשם epidural-anesthesia type
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
        "5": ["Prev. C.S. - Patient`s Request","Previous Uterine Scar"],
        "6": ["Macrosomia", "S/P Myomectomy"],
        "7": ["Fetal Thrombocytopenia","Marginal placenta","Multiple Pregnancy", "Other Indication", "Past Shoulder Dystocia", "Placenta Accreta", "Placenta previa", "Prolapse of Cord", "S/P Tear III/IV degree","Susp. Uterine Rupture", "Suspected Placental Abruption", "Suspected Vasa Previa", "TWINS PREGNANCY", "Tumor Previa", "Elderly Primipara", "Genital Herpes"]
        
    }
    update_column_with_values(data, 'surgery indication-main indication', words_dict_11, default_value="Other")
    update_column_with_values(data, 'surgery indication-secondary indication', words_dict_11, default_value="Other")
    
    ## Hysterectomy yes/no
    data = containswords_result_exists(data, 'surgery before delivery-procedure_1', ['HYSTERECTOMY'], 4, 4, 'Hysterectomy_done_yes_or_no')
    
      #עמודה בשם cs info-type of surgery
    words_dict_12 = {
        "1": ["אלקטיבי","סמי-אלקטיבי", "ניתוח קיסרי ידידותי"],
        #"2": ["סמי-אלקטיבי"],
        "3": ["דחוף", "בהול"],
        #"4": ["בהול"]
        
    }
    update_column_with_values(data, 'surgery indication-type of surgery', words_dict_12, default_value="Other", empty_value="")
    
    #create yes/no column
    data = compare_values(data, column_name='surgery indication-type of surgery', new_column_name='Elective_CS_yes/no',
                               target_value=1,
                               match_return=1,
                               no_match_return=0)
                               
    data = compare_values(data, column_name='surgery indication-type of surgery', new_column_name='Urgent_CS_yes/no',
                               target_value=3,
                               match_return=1,
                               no_match_return=0)
        
     #*עמודות YN,YR,YV,YZ - בשם Procedure
     #0-No or Hysterectomy, 1-Laparotomy, 2-Laparoscoy, 3-Other
    words_dict_13 = {
        "0": ["REPAIR","HYSTERECTOMY", "LACERATION", "UNDER"],
        "1": ["LAPAROTOMY","OPEN"],
        "2": ["LAPAROSCOP", "LAP."],
        "3": ["WIDE","DEBRIDEMENT", "DEBRIDMENT ", "INCISION AND DRAINAGE", "BREAST","HEMATOMA","OTHER", "EMBOLIZATION OF UTERINE ARTERY", "APPENDECTOMY", "colostomy", "ILEOSTOMY", "hemicolectomy"]
      
    }
    data = update_column_with_values_batch(data, 'surgery before delivery-procedure', words_dict_13, default_value="Other", empty_value="0", batch=2)
    #update_column_with_values(data, 'surgery before delivery-procedure_2', words_dict_13, default_value="Other", empty_value="0")
                                                                                                                                
    data = update_column_with_values_batch(data, 'surgery after delivery-procedure', words_dict_13, default_value="Other", empty_value="0", batch=4)
                                                                                                                                
    #update_column_with_values(data, 'surgery after delivery-procedure_2', words_dict_13, default_value="Other", empty_value="0")
    
    
    ## Remove negative values from 
    #cleared = clear_negative_values(data, '')
    #print(f"{cleared} negative \"third stage length\" values removed")

    #split estational age to week column and day column
    data = split_gestational_age(data, column_name='birth-gestational age')

    # Check if numeric values in column 'E' meet or exceed the cutoff of 1, if >= return above value, below return the below value. empty keep empty.
    data = cutoff_number(data, 'date of death - days from delivery', 'death_at_delivery_yes_no', 1, above=0, below=1, empty_value=0)
    
    # Check if numeric values in column 'T' meet or exceed the cutoff of 1, and add results in a new column
    data = cutoff_number(data, 'obstetric formula-number of cesarean sections (cs)', 'Previous_CS_yes_no', cutoff=1, above=1, below=0, empty_value='')

    # Check if numeric values in column 'X' meet or exceed the cutoff of 1, and add results in a new column
    data = cutoff_number(data, 'obstetric formula-number of vaginal births after cesarean sections (vbac)', 'Previous_VBAC_yes_no', cutoff=1, above=1, below=0, empty_value='')

    # Check if the value exists
    data = compare_values(data, 'birth-birth number', 'nulliparous_yesno', target_value=1, match_return=1, no_match_return=0)

    # Filter numbers in column 'E', removing values above 20
    data = filter_numbers(data, 'birth-pregnancy number', lowerThan=0, higherThan=20)
    
    
    #Hospital_length_of_stay_above_3d
    data = cutoff_number(data, 'hospitalization delivery-hospital length of stay', 'Hospital_length_of_stay_above_3d', 3, above=1, below=0, empty_value='')
    
    # Filter numbers in column 'AS', removing values below 15 and/or above 55
    data = filter_numbers(data, 'bmi-numeric result', lowerThan=15, higherThan=55)
    # BMI higher then 30
    data = cutoff_number(data, 'bmi-numeric result', 'BMI_above_30', 30, above=1, below=0, empty_value='')
    
     # Check if numeric values in column 'gestational age' is in-range or out-of-range for prematurity categories, and add results in a new column '___'
    data = cutoff_range_numeric(data, 'birth-gestational age', 'late_preterm_labor_yes_or_no', 34, 36.6, in_range=1, out_of_range=0, empty_value=None)
    data = cutoff_range_numeric(data, 'birth-gestational age', 'early_preterm_labor_yes_or_no', 24, 33.6, in_range=1, out_of_range=0, empty_value=None)
    #data = cutoff_number(data, 'gestational age', 'late_preterm_labor_yes_or_no', 37, above=0, below=1, empty_value='')
    #data = cutoff_number(data, 'gestational age', 'early_preterm_labor_yes_or_no', 34, above=0, below=1, empty_value='')
    
    #fever before delivery (above 38) - yes/no
    #data = cutoff_number(data, 'fever temperature numeric_max 37.5-43-numeric result', 'fever_before_delivery', 38, above=1, below=0, empty_value='') 
    
    #Apgar below 7
    data = cutoff_number(data, 'newborn sheet-apgar 1_1', 'Apgar_1m_below_7', 7, above=0, below=1, empty_value='')
    data = cutoff_number(data, 'newborn sheet-apgar 5_1', 'Apgar_5m_below_7', 7, above=0, below=1, empty_value='')
    
    # Filter numbers in coulmn 'hospitalization delivery-hospital length of stay', removing values above 100
    data = filter_numbers(data, 'hospitalization delivery-hospital length of stay', lowerThan=0, higherThan=100)

    # Flip the sign of numeric values in column 'AV' and remove values over 2500
    data = flip_sign(data, 'rom description-date of membranes rupture-hours from reference')
    data = filter_numbers(data, 'rom description-date of membranes rupture-hours from reference',lowerThan=0, higherThan=2500)
    
    # Check if the cell value in 'first antibiotics timing' is negative, and return 1 to a new column if it is. 
    #data = cutoff_number(data, 'first antibiotics timing calculated', 'Antibiotic_prophylaxis_yes_no', 0, above=0, below=1, empty_value='')
    # Check if the cell value in 'first antibiotics timing' is positive, and return 1 to a new column if it is. 
    #data = cutoff_number(data, 'first antibiotics timing calculated', 'Antibiotic_treatment_yes_no', 0, above=1, below=0, empty_value='')
    
    # Check if the cell value in 'penicillin/clindamycin timing calculated' is negative, and return 1 to a new column if it is. 
    #data = cutoff_number(data, 'penicillin/clindamycin timing calculated', 'penicillin/clindamycin_before_fever_yes_no', 0, above=0, below=1, empty_value=0)
    # Check if the cell value in 'penicillin/clindamycin timing calculated', and return 1 to a new column if it is. 
    #data = cutoff_number(data, 'penicillin/clindamycin timing calculated', 'penicillin/clindamycin_after_fever_yes_no', 0, above=1, below=0, empty_value='')
    
    # Check if the cell value in 'ampicillin timing calculated' is negative, and return 1 to a new column if it is. 
    #data = cutoff_number(data, 'ampicillin timing calculated', 'ampicillin_before_fever_yes_no', 0, above=0, below=1, empty_value=0)
    # Check if the cell value in 'ampicillin timing calculated', and return 1 to a new column if it is. 
    #data = cutoff_number(data, 'ampicillin timing calculated', 'ampicillin_after_fever_yes_no', 0, above=1, below=0, empty_value='')
    
    # Flip the sign of numeric values in column 'ANA'
    #data = flip_sign(data, 'second stage length calculated')
    
    # Multiply the values in column 'ANA' by multiplier "24",if empty stays empty. creates another column for 2nd stage longer than 4 hours. then remove values above 6. 
    data = multiply_by_number(data, 'second stage length calculated', multiplier=24)
    data = cutoff_number(data, 'second stage length calculated', 'duration_of_2nd_stage_over_4h', 4, above=1, below=0, empty_value='')

    #data = filter_numbers(data, 'second stage length', higherThan=6)
    
    #age above 35
    data = cutoff_number(data, 'birth-age when documented', 'maternal_age_above_35', 35, above=1, below=0, empty_value='')
    # Multiply the values in column 'AMZ' by multiplier "24",if empty stays empty. then remove values above 100. 
    data = multiply_by_number(data, 'length of stay delivery room calculated', multiplier=24)
    data = filter_numbers(data, 'length of stay delivery room calculated', higherThan=100)
    
    # Check if the cell value in maternal diagnosis column is empty or not, and returns 1 if its not, 0 if it is.
    data = is_empty(data, 'maternal pregestaional diabetes-diagnosis', 'maternal_pregestational_diabetes_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'maternal gestational diabetes-diagnosis', 'maternal_gestational_diabetes_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'maternal pregestational hypertension-diagnosis', 'maternal_pregestational_hypertension_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'maternal gestational hypertension-diagnosis', 'maternal_gestational_hypertension_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'maternal hellp syndrome-diagnosis', 'maternal_hellp_syndrome_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'maternal pph-diagnosis', 'maternal_pph_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'blood products given-medication', 'blood_products_given_yes_or_no', value_empty=0, value_not_empty=1)
    
  
    # Check if numeric values in column 'AV' meet or exceed the cutoff of __, and add results in a new column '___'
    data = cutoff_number(data, 'rom description-date of membranes rupture-hours from reference', 'duration_of_rom_over_18h', 18, above=1, below=0, empty_value='')

    # Flip the sign of numeric values in column 'BA'
    data = flip_sign(data, 'fever temperature numeric_max 37.5-43-date of measurement-hours from reference')

    data = combine_columns(data, ['surgery before delivery-procedure_1', 'surgery before delivery-procedure_2'], 'Surgery_before_delivery', delimiter=', ')
    data = combine_columns(data, ['surgery after delivery-procedure_1', 'surgery after delivery-procedure_2', 'surgery after delivery-procedure_3', 'surgery after delivery-procedure_4'], 'Surgery_after_delivery', delimiter=', ')

    ## CT done yes/no
    data = containswords_result_exists(data, 'imaging-exam performed (sps)_1', ['CT'], 5, 7, 'ct_done_yes_or_no')
    
    ## Drainage done yes/no
    data = containswords_result_exists(data, 'imaging-exam performed (sps)_1', ['CTI ניקוז של מורסה בבטן'], 5, 7, 'drainage_done_yes_or_no')

    # PH-Arterial
    # Clear cells in column 'AJ' and the previous column 'AI' if one of the words in the list are found
    column_list = ['ph_arterial-lab test copy(1)_1','ph_arterial-lab test copy(1)_2','ph_arterial-lab test copy(1)_3','ph_arterial-lab test copy(1)_4','ph_arterial-lab test copy(1)_5']
    words_list = ['PH-A-ST', 'PH-G-ST','PH-ST','PH-V-CORD','PH-V-ST']
    modified_data = clear_strings_multiple_columns(data, column_list, words_list, indicator=-1)
    
    data = concat_unique_values(data, ['ph_arterial-numeric result_1','ph_arterial-numeric result_2','ph_arterial-numeric result_3','ph_arterial-numeric result_4','ph_arterial-numeric result_5'], 'pH_Arteiral_Result', limitResults=1)
    
     #pH below 7.1
    data = cutoff_number(data, 'pH_Arteiral_Result', 'pH_Arteiral_below_7.1', 7.1, above=0, below=1, empty_value='')
    
    # Remove values from column 'BL' if the value in column 'BI' is equal to 0.
    #data = clear_values_based_on_reference(data, 'transfers-department length of stay', reference_column_name='transfers-department', reference_value='0')
    
     # Remove specified rows, where cultures not taken
    filtered_labor_data = remove_rows_if_contains(data, 'blood_culture_taken', ['0'])
    print(len(data)-len(filtered_labor_data), " rows without blood culture taken removed")
    data = filtered_labor_data
      
    #data = process_column_tuples(data, start_column="organisms susceptability-antibiotic_1", columns=5 ,num_tuples=65, transformations={"S": 1, "I": 2, "R": 3}, default_value=None)
    generate_heatmap_with_counts(data, start_column="organisms susceptability-antibiotic_1", columns_per_set=5 ,num_tuples=65, output_file="heatmap.csv")
    
    #filtered_newborn_data = filter_newborn_data_by_existing_patients("newborn_input.csv", data)

    data = calculate_days_between_dates(
        data,
        start_col='birth-date of first documentation - birth occurence',
        end_col='newborn sheet-hospital discharge date_1',
        result_col='Time_from_birth_to_baby_discharge_days'
    )

    data = concat_values_across_batches(data, "antibiotics-medication_1", 3, 108, "concat_antibiotics_given" )  # antibiotics-medication_1 3 X 108
    result_df = generate_patient_specific_dataset(
        data=data,
        start_column="organisms susceptability-antibiotic_1",
        columns_per_set=5,
        num_tuples=65,
        patient_id_column="patient id",
        additional_fields=["concat_antibiotics_given", "birth-gestational age", "blood_culture_organisms", "blood_culture_organisms_category", "Blood_culture_Type_of_growth"],
        output_file="patient_specific_dataset.csv"
    )

    data = add_baby_info_to_mothers(
        mothers_df=data,
        babies_file_path="newborn_input.csv",  # path to your babies file
        mother_baby_id_columns=["newborn sheet-child internal patient id_1", "newborn sheet-child internal patient id_2", "newborn sheet-child internal patient id_3"],  # adjust as needed
        babies_id_column="patient id"   # adjust as needed
    )

     # Check if the cell value in newborn diagnosis column is empty or not, and returns 1 if its not, 0 if it is.
    data = is_empty(data, 'baby_1_transfer-department', 'baby_1_NICU_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_sga-diagnosis', 'baby_1_SGA_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_lga-diagnosis', 'baby_1_LGA_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_meconium aspiration-diagnosis', 'baby_1_meconium aspiration_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_meconium -diagnosis', 'baby_1_meconium_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_hypoglycemia-diagnosis', 'baby_1_hypoglycemia_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_jaundice-diagnosis', 'baby_1_jaundice_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_rds-diagnosis', 'baby_1_RDS_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_ivh-diagnosis', 'baby_1_IVH_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_periventricular leukomalacia -diagnosis', 'baby_1_PVL_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_bronchopulmonary dysplasia -diagnosis', 'baby_1_BPD_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_necrotizing enterocolitis -diagnosis', 'baby_1_NEC_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_seizures-diagnosis', 'baby_1_seizures_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_hypoxic ischemic encephalopathy -diagnosis', 'baby_1_HIE_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_sepsis-diagnosis', 'baby_1_sepsis_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_1_mechanical ventilation-diagnosis', 'baby_1_mechanical ventilation_yes_or_no', value_empty=0, value_not_empty=1)
    
    data = add_days_between_flag(
        data,
        birth_col="baby_1_date of birth",
        death_col="baby_1_date of death",
        result_col="neonetal_death_yes_no"
    )

    data = is_empty(data, 'baby_2_transfer-department', 'baby_2_NICU_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_sga-diagnosis', 'baby_2_SGA_yes_or_no', value_empty=0, value_not_empty=1)     
    data = is_empty(data, 'baby_2_lga-diagnosis', 'baby_2_LGA_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_meconium aspiration-diagnosis', 'baby_2_meconium aspiration_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_meconium -diagnosis', 'baby_2_meconium_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_hypoglycemia-diagnosis', 'baby_2_hypoglycemia_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_jaundice-diagnosis', 'baby_2_jaundice_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_rds-diagnosis', 'baby_2_RDS_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_ivh-diagnosis', 'baby_2_IVH_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_periventricular leukomalacia -diagnosis', 'baby_2_PVL_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_bronchopulmonary dysplasia -diagnosis', 'baby_2_BPD_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_necrotizing enterocolitis -diagnosis', 'baby_2_NEC_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_seizures-diagnosis', 'baby_2_seizures_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_hypoxic ischemic encephalopathy -diagnosis', 'baby_2_HIE_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_mechanical ventilation-diagnosis', 'baby_2_mechanical ventilation_yes_or_no', value_empty=0, value_not_empty=1)
    data = is_empty(data, 'baby_2_sepsis-diagnosis', 'baby_2_sepsis_yes_or_no', value_empty=0, value_not_empty=1)
    
    #data = is_empty(data, 'baby_3_transfer-department', 'baby_3_NICU_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_sga-diagnosis', 'baby_3_SGA_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_lga-diagnosis', 'baby_3_LGA_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_meconium aspiration-diagnosis', 'baby_3_meconium aspiration_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_meconium -diagnosis', 'baby_3_meconium_yes_or_no', value_empty=0, value_not_empty=1) 
    #data = is_empty(data, 'baby_3_hypoglycemia-diagnosis', 'baby_3_hypoglycemia_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_jaundice-diagnosis', 'baby_3_jaundice_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_rds-diagnosis', 'baby_3_RDS_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_ivh-diagnosis', 'baby_3_IVH_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_periventricular leukomalacia -diagnosis', 'baby_3_PVL_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_bronchopulmonary dysplasia -diagnosis', 'baby_3_BPD_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_necrotizing enterocolitis -diagnosis', 'baby_3_NEC_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_seizures-diagnosis', 'baby_3_seizures_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_hypoxic ischemic encephalopathy -diagnosis', 'baby_3_HIE_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_sepsis-diagnosis', 'baby_3_sepsis_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_3_mechanical ventilation-diagnosis', 'baby_3_mechanical ventilation_yes_or_no', value_empty=0, value_not_empty=1)

    data = time_to_treatment_after_event(
        data,
        event_date_col='onset of fever 38 until delivery-date of measurement',
        abx_med_col='antibiotics-medication_1',
        abx_date_col='antibiotics-date administered _1',
        num_abx=108,
        step_size=3,
        result_col='time_from_intrapartum_fever_treatment_hours'
    )
    
    # ABX from list given before "onset of fever 38 before delivery", or before delivery if there was no "onset of fever 38 before delivery" 
   # data = flag_antibiotic_within_timeframe_idx(
    #    data, 'onset of fever 38 until delivery-date of measurement', 'antibiotics-medication_1', 'antibiotics-date administered _1', 103, 3, -96, 'Penicillin/Clindamycin_given_in_48h_before_fever', antibiotics_to_include=['PENICILLIN G SODIUM', 'CLINDAMYCIN HCL', 'CLINDAMYCIN PHOSPHATE', 'AMPICILLIN', 'AMOXYCILLIN'], alternative_event_date_col="birth-date of first documentation - birth occurence"
    #)
    

    
    # Example 1: Any abx within 48 hours *before* event
    #data = flag_antibiotic_within_timeframe_idx(
        #data, 'onset of fever 38 after delivery-date of measurement', 'antibiotics-medication_1', 'antibiotics-date administered _1', 103, 3, -48, 'abx_given_in_48h_before_fever'
    #)
    #data = flag_antibiotic_within_timeframe_idx(
        #data, 'onset of fever 38 after delivery-date of measurement', 'antibiotics-medication_1', 'antibiotics-date administered _1', 103, 3, 'birth-date of first documentation - birth occurence', 'abx_given_between_delivery_and_fever'
    #)
    #data = flag_antibiotic_within_timeframe_idx(
       #data, 'onset of fever 38 after delivery-date of measurement', 'antibiotics-medication_1', 'antibiotics-date administered _1', 103, 3, 96, 'abx_given_up_to_96h_after_fever'
    #)

    # Example #3: Only Ceftriaxone given between delivery and first fever
    
    data = flag_antibiotic_within_timeframe_idx(
        data, 'onset of fever 38 until delivery-date of measurement', 'antibiotics-medication_1', 'antibiotics-date administered _1', 108, 3, 'birth-date of first documentation - birth occurence', 'GBS_prophylactic_treatment_yes/no', antibiotics_to_include=['PENICILLIN G SODIUM', 'CLINDAMYCIN HCL', 'CLINDAMYCIN PHOSPHATE']
    )
    #data = flag_antibiotic_within_timeframe_idx(
        #data, 'onset of fever 38 after delivery-date of measurement', 'antibiotics-medication_1', 'antibiotics-date administered _1', 103, 3, 96, 'Penicillin/Clindamycin_given_up_to_96h_after_fever', antibiotics_to_include=['PENICILLIN G SODIUM', 'CLINDAMYCIN HCL', 'CLINDAMYCIN PHOSPHATE']
    #)

    #data = flag_antibiotic_within_timeframe_idx(
        #data, 'onset of fever 38 after delivery-date of measurement', 'antibiotics-medication_1', 'antibiotics-date administered _1', 103, 3, -48, 'Ampicillin_given_in_48h_before_fever', antibiotics_to_include=['AMPICILLIN']
    #)
    #data = flag_antibiotic_within_timeframe_idx(
        #data, 'onset of fever 38 after delivery-date of measurement', 'antibiotics-medication_1', 'antibiotics-date administered _1', 103, 3, 'birth-date of first documentation - birth occurence', 'Ampicillin_given_between_delivery_and_fever', antibiotics_to_include=['AMPICILLIN']
    #)
    #data = flag_antibiotic_within_timeframe_idx(
        #data, 'onset of fever 38 after delivery-date of measurement', 'antibiotics-medication_1', 'antibiotics-date administered _1', 103, 3, 96, 'Ampicillin_given_up_to_96h_after_fever', antibiotics_to_include=['AMPICILLIN']
    #)


    data_split = split_twin_rows(
        df=data,
        b1b1_first_col='baby1_weight', b1b1_num_cols=18, b1b1_rm_prefix='baby1_',
        b1b2_first_col='score1_apgar', b1b2_num_cols=8, b1b2_rm_prefix='score1_',
        b2b1_first_col='baby2_weight', b2b1_num_cols=18, b2b1_rm_prefix='baby2_',
        b2b2_first_col='score2_apgar', b2b2_num_cols=8, b2b2_rm_prefix='score2_',
        b3b1_first_col=None, b3b1_num_cols=None, b3b1_rm_prefix=None,
        b3b2_first_col=None, b3b2_num_cols=None, b3b2_rm_prefix=None
    )

    # Remove specified columns, including single columns and ranges
    columns_to_remove = [
        'reference occurrence number',
        'date of birth~date of death - days from delivery',
        'birth-date of first documentation - birth occurence',
        'hospitalization delivery-hospital admission date',
        'hospitalization delivery-hospital discharge date',
        'newborn sheet-hospital admission date_1',
        'newborn sheet-hospital discharge date_1',
        #'newborn sheet-apgar 1_2~newborn sheet-child internal patient id_2',
        'newborn sheet-apgar 1_3~newborn sheet-child internal patient id_3',
        'ph_arterial-numeric result_1~ph_arterial-lab test copy(1)_5',
        'bmi-date of measurement-days from reference',
        'second and third stage timeline-time of full dilation',
        'gbs status-gbs in urine',
        'gbs status-gbs vagina',
        'fever temperature numeric_max 37.5-43-date of measurement',
        #'onset of fever 38 until delivery-date of measurement-hours from reference',
        'onset of fever 38 until delivery-date of measurement',
        'wbc max-collection date-hours from reference',
        'crp max-collection date-hours from reference',
        'transfers-department admission date-days from reference',
        'transfers-department discharge date-days from reference',
        'readmission-hospital admission date-days from reference',
        'readmission-hospital admission date',
        'readmission-hospital discharge date-days from reference',
        'readmission-hospital discharge date',
        'cultures-test type_1~cultures-stain_61',
        'organisms susceptability-antibiotic_1~organisms susceptability-antibiotic panel_65',
        'surgery before delivery-date of procedure-days from reference_1~surgery after delivery-department_4',
        'antibiotics_first-date administered-hours from reference~antibiotics_first-medication',
        'length of stay-room name~length of stay-room exit date',
        'imaging-exam performed (sps)_1~imaging-performed procedures_7',
        'antibiotics-date administered-hours from reference_1~antibiotics-medication_108',
        'surgery indication-type of surgery',
        'obstetric formula-number of abortions (ab)',
        'obstetric formula-number of births (p)',
        'obstetric formula-number of ectopic pregnancies (eup)',
        'obstetric formula-number of live children (lc)',
        'obstetric formula-number of pregnancies (g)',
        'maternal pregestaional diabetes-diagnosis',    
        'maternal gestational diabetes-diagnosis',  
        'maternal pregestational hypertension-diagnosis',   
        'maternal gestational hypertension-diagnosis',  
        'maternal hellp syndrome-diagnosis',    
        'maternal pph-diagnosis',
        'blood products given-medication',
        'penicillin clindamycin prophylaxis-date administered-hours from reference',
        'penicillin clindamycin prophylaxis-date administered', 
        'penicillin clindamycin prophylaxis-medication',    
        'ampicillin prophylaxis-date administered-hours from reference',    
        'ampicillin prophylaxis-date administered', 
        'ampicillin prophylaxis-medication',
        'penicillin/clindamycin timing calculated', 
        'ampicillin timing calculated',
        'first antibiotics timing calculated',
        'blood_culture_taken',
        'birth-child internal patient id',
        'baby_1_date of birth',
        'baby_1_date of death',
        'baby_1_reference occurrence number',
        'baby_1_cohort reference event-age_at_event',
        'baby_1_neonatal death -diagnosis~baby_1_death2-weight',
        'baby_1_apgar -apgar 1~baby_1_apgar -apgar 10',
        'baby_1_apgar -weight~baby_1_apgar -pregnancy age',
        'baby_1_apgar -department~baby_1_apgar -transferred to intensive care',
        'baby_1_apgar -documented birth time-days from reference~baby_1_apgar -patient id of newborn',
        'baby_1_sga-diagnosis~baby_1_mechanical ventilation-diagnosis',
        'baby_1_apgar -method of delivery',
        'baby_1_transfer-department',
        'baby_1_apgar -birth rescue',
        'baby_1_apgar -birth experience',
        'baby_1_apgar -child number (in current delivery)',
        'baby_2_date of birth',
        'baby_2_reference occurrence number~baby_3_apgar -patient id of newborn'
        #'baby_2_NICU_yes_or_no~baby_2_mechanical ventilation_yes_or_no',
        #'baby_3_NICU_yes_or_no~baby_3_mechanical ventilation_yes_or_no'
    ]
    data = remove_columns(data, columns_to_remove)
    data_split = remove_columns(data_split, columns_to_remove)

    #data = add_row_index_column(data, col_name="Index")
    data = replace_column_spaces(data)
    data_split = replace_column_spaces(data_split)

    print(f' Data length: {len(data)}, Split twin data length: {len(data_split)}')
    save_data(data, 'output.csv')
    save_data(data_split, 'output_twins_split.csv')
    #split_and_save_csv(data, 'fever temperature numeric_max 37.5-43-numeric result', 'output.csv', 'output_under_38.csv', 'output_38_or_above.csv', encoding='utf-8')

if __name__ == "__main__":
    main()