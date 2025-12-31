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
    match = re.match(r'^([A-Z][A-Za-z+]*(?: [A-Z]+)*)', name)
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
                concatenated_results = [dictionary.get(item.strip(), "Uncategorized") for item in concatenated_results]

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
    Leaves empty cells as empty in the new column.

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

    
def reorder_columns(data, ordered_cols):
    """
    Reorders the DataFrame columns so that the specified ordered_cols appear first (in order),
    followed by all other columns in their original order.

    Args:
        data: pandas DataFrame
        ordered_cols: list of column names to move to the front (in this order)

    Returns:
        DataFrame with reordered columns
    """
    existing_ordered = [col for col in ordered_cols if col in data.columns]
    remaining = [col for col in data.columns if col not in existing_ordered]
    new_order = existing_ordered + remaining
    return data[new_order]

def add_row_index_column(data, col_name="Index", first_position=True):
    """
    Adds a 1-based row index column to the DataFrame.
    If first_position is True, inserts it as the first column.
    """
    indexed = data.copy()
    indexed[col_name] = range(1, len(indexed) + 1)
    
    if first_position:
        cols = [col_name] + [col for col in indexed.columns if col != col_name]
        indexed = indexed[cols]
    
    return indexed


def count_unique_growths_by_dfr(
    data,
    dfr_first_col,              # e.g., "blood cultures-date collected-days from reference_1"
    growth_first_col,           # e.g., "blood cultures-organism detected_1"
    num_steps,
    step_size,
    organism_dict,
    map_column_name=None,       # e.g., "growth_counts_map"
    prefix_for_columns=None     # e.g., "growth_"
):
    """
    For each row: scan batched (DaysFromReference, growth), split growth on ';',
    map each token via organism_dict, dedupe by (DaysFromReference, mapped_growth), count per mapped_growth.
    Optionally adds a compact map column and/or wide per-growth _count columns.
    """
    import re
    from collections import defaultdict

    out = data.copy()

    days_base_index = column_name_to_index(out, dfr_first_col)
    growth_base_index = column_name_to_index(out, growth_first_col)

    def _safe_col(s):
        return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")

    all_growth_labels = set()
    per_row_counts = []

    for row_index, row in out.iterrows():
        seen_pairs = set()
        counts = defaultdict(int)

        for step in range(num_steps):
            offset = step * step_size
            days_index = days_base_index + offset
            growth_index = growth_base_index + offset
            if days_index >= len(row) or growth_index >= len(row):
                continue

            # days-from-reference (float)
            raw_days = row.iloc[days_index]
            days_value = pd.to_numeric(raw_days, errors="coerce")

            raw_growth_cell = row.iloc[growth_index]
            if pd.isna(raw_growth_cell):
                continue

            # split multi-organism cell on ';'
            for token in str(raw_growth_cell).split(';'):
                token = token.strip()
                if not token:
                    continue

                mapped_growth = organism_dict.get(token)
                if not mapped_growth:  # skip unknowns; change to get(token, token) to include them
                    print(f"Error finding organizm category for {token}")
                    mapped_growth = organism_dict.get(token, token)

                if pd.isna(days_value):
                    print(f"[growth-count] row={row_index}: growth '{mapped_growth}' has no valid DaysFromReference at batch {step}")
                    continue

                pair_key = (float(days_value), mapped_growth)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                counts[mapped_growth] += 1

        per_row_counts.append(dict(counts))
        if prefix_for_columns:
            # only keep valid string labels to avoid sort issues
            all_growth_labels.update([g for g in counts.keys() if isinstance(g, str) and g])

    # wide columns (one per growth)
    if prefix_for_columns:
        ordered_growths = sorted(all_growth_labels, key=custom_sort)
        for g in ordered_growths:
            out[f"{prefix_for_columns}{_safe_col(g)}_count"] = 0

        for i, counts in enumerate(per_row_counts):
            ridx = out.index[i]
            for g, c in counts.items():
                out.at[ridx, f"{prefix_for_columns}{_safe_col(g)}_count"] = c

    # compact map column like "E. coli:2; Staph aureus:1"
    if map_column_name:
        maps = []
        for counts in per_row_counts:
            if counts:
                keys = [g for g in counts.keys() if isinstance(g, str) and g]
                parts = [f"{g}:{counts[g]}" for g in sorted(keys, key=custom_sort)]
                maps.append("; ".join(parts))
            else:
                maps.append("")
        out[map_column_name] = maps

    return out


def evaluate_appropriate_antibiotic_treatment(
    originalData,
    # Antibiotic columns (time in HOURS from reference)
    abx_med_col="antibiotics-medication_1",
    abx_time_col="antibiotics-date administered-hours from reference_1",
    abx_step_size=3,
    abx_num_steps=108,
    # Growth (culture) columns (time in DAYS from reference)
    growth_org_col="cultures-organism detected_1",
    growth_time_col="cultures-collection date-days from reference_1",
    growth_step_size=8,
    growth_num_steps=61,
    # Susceptibility columns
    susc_abx_col="organisms susceptability-antibiotic_1",
    susc_micro_col="organisms susceptability-microorganism_1",
    susc_interp_col="organisms susceptability-susceptibility interpretation_1",
    susc_step_size=5,
    susc_num_steps=65,
    # Time window in HOURS, centered on the GROWTH time
    timeframe_before_hours=10.0,   # hours before growth
    timeframe_after_hours=10.0,    # hours after growth
    # Output column names
    raw_output_col="abx_growth_susceptibility_raw",
    per_growth_output_col="abx_growth_susceptibility_by_growth",
    overall_output_col="abx_appropriate_treatment_overall"
):
    """
    Growth-centric evaluation of appropriate antibiotic treatment.
    """

    data = originalData.copy()

    # Resolve base indices (allow passing either column names or integer indices)
    abx_med_start = column_name_to_index(data, abx_med_col) if isinstance(abx_med_col, str) else abx_med_col
    abx_time_start = column_name_to_index(data, abx_time_col) if isinstance(abx_time_col, str) else abx_time_col

    growth_org_start = column_name_to_index(data, growth_org_col) if isinstance(growth_org_col, str) else growth_org_col
    growth_time_start = column_name_to_index(data, growth_time_col) if isinstance(growth_time_col, str) else growth_time_col

    susc_abx_start = column_name_to_index(data, susc_abx_col) if isinstance(susc_abx_col, str) else susc_abx_col
    susc_micro_start = column_name_to_index(data, susc_micro_col) if isinstance(susc_micro_col, str) else susc_micro_col
    susc_interp_start = column_name_to_index(data, susc_interp_col) if isinstance(susc_interp_col, str) else susc_interp_col

    def _normalize_microorganism(name):
        return str(name).strip().upper()

    def _parse_numeric(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        if s == "":
            return None
        try:
            return float(s)
        except Exception:
            return None

    def _combine_status(prev_status, new_status):
        # Priority: S (best) > I > R > X (unknown)
        if prev_status is None:
            return new_status
        priority = {"S": 3, "I": 2, "R": 1, "X": 0}
        prev_rank = priority.get(prev_status, -1)
        new_rank = priority.get(new_status, -1)
        return prev_status if prev_rank >= new_rank else new_status

    raw_results_all_rows = []
    per_growth_results_all_rows = []
    overall_results_all_rows = []

    for _, row in data.iterrows():
        # 1) Collect growth events, splitting multi-organism strings by ';'
        growth_events = []  # { "org_raw", "org_key", "time_hours" }
        for step in range(growth_num_steps):
            org_col_idx = growth_org_start + step * growth_step_size
            time_col_idx = growth_time_start + step * growth_step_size

            if org_col_idx >= len(data.columns) or time_col_idx >= len(data.columns):
                break

            org_val = row.iloc[org_col_idx]
            if pd.isna(org_val):
                continue

            org_str = str(org_val).strip()
            if org_str == "":
                continue

            time_days = _parse_numeric(row.iloc[time_col_idx])
            if time_days is None:
                continue
            time_hours = time_days * 24.0  # days -> hours

            # Split by ';' – each organism is a separate growth entry
            org_parts = [p.strip() for p in org_str.split(";") if p.strip() != ""]
            for org_part in org_parts:
                org_key = _normalize_microorganism(org_part)
                growth_events.append({
                    "org_raw": org_part,
                    "org_key": org_key,
                    "time_hours": time_hours
                })

        # 2) Collect antibiotic events (time already in HOURS)
        abx_events = []  # { "abx_raw", "abx_key", "time_hours" }
        for step in range(abx_num_steps):
            med_col_idx = abx_med_start + step * abx_step_size
            time_col_idx = abx_time_start + step * abx_step_size

            if med_col_idx >= len(data.columns) or time_col_idx >= len(data.columns):
                break

            med_val = row.iloc[med_col_idx]
            if pd.isna(med_val) or str(med_val).strip() == "":
                continue

            time_hours = _parse_numeric(row.iloc[time_col_idx])
            if time_hours is None:
                continue

            abx_key = _sanitize_antibiotic_name(med_val)
            if abx_key == "":
                continue

            abx_events.append({
                "abx_raw": med_val,
                "abx_key": abx_key,
                "time_hours": time_hours
            })

        # 3) Build susceptibility map: (abx_key, org_key) -> [S/I/R]
        susc_map = defaultdict(list)
        for step in range(susc_num_steps):
            abx_col_idx = susc_abx_start + step * susc_step_size
            micro_col_idx = susc_micro_start + step * susc_step_size
            interp_col_idx = susc_interp_start + step * susc_step_size

            if interp_col_idx >= len(data.columns):
                break

            abx_val = row.iloc[abx_col_idx]
            micro_val = row.iloc[micro_col_idx]
            interp_val = row.iloc[interp_col_idx]

            if pd.isna(abx_val) or str(abx_val).strip() == "":
                continue
            if pd.isna(micro_val) or str(micro_val).strip() == "":
                continue
            if pd.isna(interp_val) or str(interp_val).strip() == "":
                continue

            abx_key = _sanitize_antibiotic_name(abx_val)
            if abx_key == "":
                continue

            micro_key = _normalize_microorganism(micro_val)
            interp = str(interp_val).strip().upper()
            if interp not in ("S", "I", "R"):
                continue

            susc_map[(abx_key, micro_key)].append(interp)

        # 4) Match growths to antibiotics within timeframe, build pairs
        raw_pairs_this_row = []
        seen_raw_pairs = set()        # (abx_raw, org_raw, status)
        per_growth_status = {}        # org_key -> S/I/R/X

        for g in growth_events:
            g_time = g["time_hours"]

            for a in abx_events:
                delta_hours = a["time_hours"] - g_time
                if delta_hours < -timeframe_before_hours or delta_hours > timeframe_after_hours:
                    continue

                key = (a["abx_key"], g["org_key"])
                interps = susc_map.get(key, [])

                if interps:
                    # Aggregate S/I/R for this pair: S > I > R
                    if any(v == "S" for v in interps):
                        pair_status = "S"
                    elif any(v == "I" for v in interps):
                        pair_status = "I"
                    else:
                        pair_status = "R"
                else:
                    # No susceptibility info: keep the pair with status "X"
                    pair_status = "X"

                raw_tuple = (str(a["abx_raw"]).strip(), str(g["org_raw"]).strip(), pair_status)
                if raw_tuple not in seen_raw_pairs:
                    seen_raw_pairs.add(raw_tuple)
                    raw_pairs_this_row.append(
                        f"{raw_tuple[0]} | {raw_tuple[1]} | {raw_tuple[2]}"
                    )

                prev_status = per_growth_status.get(g["org_key"])
                per_growth_status[g["org_key"]] = _combine_status(prev_status, pair_status)

        # 5) Build outputs

        # 5a) Raw pairs
        if raw_pairs_this_row:
            raw_results_all_rows.append("; ".join(raw_pairs_this_row))
        else:
            raw_results_all_rows.append("")

        # 5b) Per-growth flags and 5c) overall 1/0/""
        if per_growth_status:
            ordered_org_keys = []
            seen_orgs = set()
            for g in growth_events:
                ok = g["org_key"]
                if ok in per_growth_status and ok not in seen_orgs:
                    ordered_org_keys.append(ok)
                    seen_orgs.add(ok)

            per_growth_flags = [per_growth_status[ok] for ok in ordered_org_keys]
            per_growth_results_all_rows.append(",".join(per_growth_flags))

            # Overall: 1 if ALL are S, else 0
            if all(flag == "S" for flag in per_growth_flags):
                overall_results_all_rows.append("1")
            else:
                overall_results_all_rows.append("0")
        else:
            per_growth_results_all_rows.append("")
            overall_results_all_rows.append("")

    data[raw_output_col] = raw_results_all_rows
    data[per_growth_output_col] = per_growth_results_all_rows
    data[overall_output_col] = overall_results_all_rows

    # ============================================================
    # =============== NEW MINIMAL ADDITION BELOW =================
    # ============================================================

    # Parse each row's raw triplets into grouped structure
    parsed_by_row = []
    max_growths = 0
    max_abx = 0   # track max antibiotics per growth

    for cell in data[raw_output_col].fillna("").astype(str):
        if cell.strip() == "":
            parsed_by_row.append(([], []))
            continue

        trips = [t.strip() for t in cell.split(";") if t.strip()]
        grouped = {}
        order = []

        for t in trips:
            parts = [p.strip() for p in t.split("|")]
            if len(parts) != 3:
                continue
            abx_raw, org_raw, status_raw = parts
            status_raw = status_raw.upper()
            disp = "N/A" if status_raw == "X" else status_raw

            if org_raw not in grouped:
                grouped[org_raw] = []
                order.append(org_raw)

            grouped[org_raw].append((abx_raw, disp))

        names = order
        lists = [grouped[g] for g in order]

        max_growths = max(max_growths, len(names))
        for abx_list in lists:
            max_abx = max(max_abx, len(abx_list))

        parsed_by_row.append((names, lists))

    # Status → numeric mapping
    status_to_num = {"S": 0, "I": 1, "R": 2, "N/A": 3}

    # Create dynamic columns:
    # growth_i_name
    # growth_i_abx_k
    # growth_i_abx_k_status
    # growth_i_abx_k_status_numeric  <-- NEW

    for i in range(max_growths):
        name_col = []

        abx_cols = [[] for _ in range(max_abx)]
        status_cols = [[] for _ in range(max_abx)]
        numeric_cols = [[] for _ in range(max_abx)]   # NEW

        for (names, lists) in parsed_by_row:
            if i < len(names):
                name_val = names[i]
                abx_list = lists[i]

                for k in range(max_abx):
                    if k < len(abx_list):
                        abx_raw, status_raw = abx_list[k]
                        abx_cols[k].append(abx_raw)
                        status_cols[k].append(status_raw)
                        numeric_cols[k].append(status_to_num.get(status_raw, 3))
                    else:
                        abx_cols[k].append("")
                        status_cols[k].append("")
                        numeric_cols[k].append("")
            else:
                name_val = ""
                for k in range(max_abx):
                    abx_cols[k].append("")
                    status_cols[k].append("")
                    numeric_cols[k].append("")

            name_col.append(name_val)

        data[f"growth_{i+1}_name"] = name_col

        for k in range(max_abx):
            data[f"growth_{i+1}_abx_{k+1}"] = abx_cols[k]
            data[f"growth_{i+1}_abx_{k+1}_status"] = status_cols[k]
            data[f"growth_{i+1}_abx_{k+1}_status_numeric"] = numeric_cols[k]   # NEW

    # ============================================================
    # =============== END OF MINIMAL ADDITION ====================
    # ============================================================

    print(
        f"Columns '{raw_output_col}', '{per_growth_output_col}', and "
        f"'{overall_output_col}' added (growth-centric appropriate treatment evaluation)."
        f"Dynamic abx_growth_N_* columns also created."
    )
    return data

def create_growth_centric_dataset_smart_v3(data, output_file="growth_centric_smart_analysis_v3.csv"):
    """
    SMART Version V3:
    - Row per Growth.
    - Handles Multiple Susceptibility Tests (Columns: _Res, _Res_test2...).
    - **TIME FILTERING**: Matches Antibiotics to Growth only if given within [-10h, +72h] of culture.
    """
    print("Starting generation of SMART Growth-Centric Dataset V3 (Time-Filtered)...")

    # 1. Context Columns (Demographics)
    keep_cols = [
        #"patient id", "birth-age_when_documented", "birth-gestational_age", 
        #"maternal_age_above_35", "fever_temperature_numeric_max_37.5-43-numeric_result",
        #"birth-type_of_labor_onset", "birth-birth_site", "surgery_indication-type_of_surgery",
        #"time_from_intrapartum_fever_treatment_hours"
        "patient id"
    ]
    base_cols = [c for c in keep_cols if c in data.columns]

    # 2. Batch Indices & Parameters
    
    # A. Growth (61 steps, 8 cols each)
    # We need the numeric date to calculate the window
    growth_name_idx = column_name_to_index(data, "cultures-organism detected_1")
    growth_date_idx = column_name_to_index(data, "cultures-collection date-days from reference_1") # DAYS
    growth_steps, growth_step_size = 61, 8
    
    # B. Susceptibility (65 steps, 5 cols each)
    susc_abx_idx = column_name_to_index(data, "organisms susceptability-antibiotic_1")
    susc_org_idx = column_name_to_index(data, "organisms susceptability-microorganism_1")
    susc_res_idx = column_name_to_index(data, "organisms susceptability-susceptibility interpretation_1")
    susc_steps, susc_step_size = 65, 5
    
    # C. Treatments Given (108 steps, 3 cols each)
    # We need the numeric hours to compare with growth time
    tx_med_idx = column_name_to_index(data, "antibiotics-medication_1")
    tx_date_str_idx = column_name_to_index(data, "antibiotics-date administered _1")
    tx_date_hrs_idx = column_name_to_index(data, "antibiotics-date administered-hours from reference_1") # HOURS
    tx_steps, tx_step_size = 108, 3

    # Timeframe Window (Hours)
    WINDOW_PRE = 10   # 10 hours before
    WINDOW_POST = 72  # 72 hours after

    new_rows = []
    
    # Stats
    stats_patients_with_growth = set()
    stats_abx_relevant_count = [] 
    stats_max_susc_multiplicity = 1 

    for idx, row in data.iterrows():
        patient_id = row["patient id"]

        # --- A. Pre-load ALL Treatments for this Patient ---
        # List of tuples: (Normalized_Name, Date_String, Hours_Float)
        patient_treatments = []
        
        for i in range(tx_steps):
            m_idx = tx_med_idx + (i * tx_step_size)
            d_str_idx = tx_date_str_idx + (i * tx_step_size)
            d_hrs_idx = tx_date_hrs_idx + (i * tx_step_size)
            
            if m_idx >= len(row): break
            
            raw_med = row.iloc[m_idx]
            if pd.notna(raw_med) and str(raw_med).strip() not in ["", "nan", "None", "0"]:
                
                # Get Timing
                try:
                    hrs_val = float(row.iloc[d_hrs_idx])
                except (ValueError, TypeError):
                    hrs_val = None # Cannot use for time filtering if missing numeric time
                
                date_str = str(row.iloc[d_str_idx]).strip()
                norm_name = _sanitize_antibiotic_name_v2(raw_med)
                
                if norm_name:
                    patient_treatments.append({
                        "name": norm_name,
                        "date_str": date_str,
                        "hours": hrs_val
                    })

        # --- B. Build Susceptibility Map (Handle Multiples) ---
        susc_map = defaultdict(lambda: defaultdict(list))
        for i in range(susc_steps):
            o_idx = susc_org_idx + (i * susc_step_size)
            a_idx = susc_abx_idx + (i * susc_step_size)
            r_idx = susc_res_idx + (i * susc_step_size)
            if r_idx >= len(row): break
            
            s_org = row.iloc[o_idx]
            raw_abx = row.iloc[a_idx]
            s_res = row.iloc[r_idx]
            
            if pd.notna(s_org) and pd.notna(raw_abx) and pd.notna(s_res):
                s_org_str = str(s_org).strip().upper()
                raw_abx_str = str(raw_abx).strip()
                s_res_str = str(s_res).strip().upper()

                if s_org_str and raw_abx_str and s_res_str not in ["", "NAN", "NONE"]:
                    norm_abx = _sanitize_antibiotic_name_v2(raw_abx_str)
                    susc_map[s_org_str][norm_abx].append(s_res_str)

        # --- C. Iterate Growths ---
        patient_had_growth = False
        
        for i in range(growth_steps):
            g_name_i = growth_name_idx + (i * growth_step_size)
            g_date_i = growth_date_idx + (i * growth_step_size)
            if g_name_i >= len(row): break
            
            growth_name = row.iloc[g_name_i]
            growth_days_val = row.iloc[g_date_i] # This is in DAYS
            
            # Check Valid Growth
            if pd.notna(growth_name) and str(growth_name).strip() not in ["", "nan", "None", "0"]:
                growth_name_str = str(growth_name).strip()
                
                # Calculate Growth Time in Hours
                growth_hours = None
                try:
                    growth_hours = float(growth_days_val) * 24.0
                except (ValueError, TypeError):
                    growth_hours = None # Cannot filter accurately if missing time
                
                patient_had_growth = True
                
                # --- Filter Treatments for THIS Growth ---
                # We want a map: {Abx_Name: Date_String} 
                # BUT only if the Abx time is within [growth - 10, growth + 72]
                relevant_tx_map = {}
                count_relevant = 0
                
                if growth_hours is not None:
                    window_start = growth_hours - WINDOW_PRE
                    window_end = growth_hours + WINDOW_POST
                    
                    for tx in patient_treatments:
                        if tx["hours"] is not None:
                            # Strict Time Filter
                            if window_start <= tx["hours"] <= window_end:
                                # Found a match!
                                if tx["name"] not in relevant_tx_map:
                                    relevant_tx_map[tx["name"]] = tx["date_str"]
                                    count_relevant += 1
                        else:
                            # Fallback: If abx has no time, do we include it? 
                            # Usually safest to exclude to avoid noise, or include with flag.
                            # For strict analysis, we exclude.
                            pass
                else:
                    # If Growth has no date, we can't filter. 
                    # Decision: Include nothing (safest) or everything? 
                    # Assuming data is clean, we skip.
                    pass

                stats_abx_relevant_count.append(count_relevant)

                # --- Create Row ---
                new_row = {col: row[col] for col in base_cols}
                new_row['Target_Growth_Name'] = growth_name_str
                # Keep original days string/val
                new_row['Target_Growth_Date_Days'] = str(growth_days_val) 
                
                # Get susceptibility results
                my_susc_dict = susc_map.get(growth_name_str.upper(), {})
                
                for abx_norm, results_list in my_susc_dict.items():
                    if len(results_list) > stats_max_susc_multiplicity:
                        stats_max_susc_multiplicity = len(results_list)
                    
                    # Susceptibility Results
                    new_row[f"{abx_norm}_Res"] = results_list[0]
                    for k in range(1, len(results_list)):
                        new_row[f"{abx_norm}_Res_test{k+1}"] = results_list[k]
                    
                    # Date Given (Only if in relevant_tx_map)
                    date_given = relevant_tx_map.get(abx_norm, "")
                    new_row[f"{abx_norm}_Date_Given"] = date_given
                
                new_rows.append(new_row)
        
        if patient_had_growth:
            stats_patients_with_growth.add(patient_id)

    # --- D. Finalize & Stats ---
    output_df = pd.DataFrame(new_rows)
    
    # Calc Stats
    num_growths = len(output_df)
    num_patients = len(stats_patients_with_growth)
    
    if stats_abx_relevant_count:
        min_abx = min(stats_abx_relevant_count)
        max_abx = max(stats_abx_relevant_count)
        avg_abx = sum(stats_abx_relevant_count) / len(stats_abx_relevant_count)
    else:
        min_abx, max_abx, avg_abx = 0, 0, 0

    print("="*50)
    print("      SMART ANALYSIS STATISTICS (V3)      ")
    print(f"      Filter Window: -{WINDOW_PRE}h to +{WINDOW_POST}h")
    print("="*50)
    print(f"Total Growth Rows: {num_growths}")
    print(f"Unique Patients with Growth: {num_patients}")
    print("-" * 30)
    print("Relevant Antibiotics (within timeframe) per Growth:")
    print(f"  Min: {min_abx}")
    print(f"  Max: {max_abx}")
    print(f"  Avg: {avg_abx:.2f}")
    print("-" * 30)
    print(f"Max Susceptibility Tests for single Abx: {stats_max_susc_multiplicity}")
    print("="*50)

    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Smart Growth-Centric dataset saved to {output_file}")
    return output_df


def create_growth_centric_dataset_smart_v9(data, output_file="growth_centric_smart_analysis_final.csv"):
    """
    SMART Version V9 (No Dictionary, No Matching):
    1. Row per Growth.
    2. TREATMENT COLUMNS: Scans [-10h, +72h] window.
       - Copies RAW Name, Date, and Hours for every valid entry.
    3. SUSCEPTIBILITY COLUMNS: Lists all lab results independently.
    """
    print("Starting generation of SMART Growth-Centric Dataset V9 (Pure Time Filter)...")

    # 1. Context Columns
    keep_cols = [
        "patient id"
    ]
    base_cols = [c for c in keep_cols if c in data.columns]

    # 2. Batch Indices
    growth_name_idx = column_name_to_index(data, "cultures-organism detected_1")
    growth_date_idx = column_name_to_index(data, "cultures-collection date-days from reference_1")
    growth_steps, growth_step_size = 61, 8
    
    susc_abx_idx = column_name_to_index(data, "organisms susceptability-antibiotic_1")
    susc_org_idx = column_name_to_index(data, "organisms susceptability-microorganism_1")
    susc_res_idx = column_name_to_index(data, "organisms susceptability-susceptibility interpretation_1")
    susc_steps, susc_step_size = 65, 5
    
    tx_med_idx = column_name_to_index(data, "antibiotics-medication_1")
    tx_date_str_idx = column_name_to_index(data, "antibiotics-date administered _1") 
    tx_date_hrs_idx = column_name_to_index(data, "antibiotics-date administered-hours from reference_1") 
    tx_steps, tx_step_size = 108, 3

    # Timeframe Window (Hours)
    WINDOW_PRE = 10 
    WINDOW_POST = 10

    new_rows = []
    
    # Stats
    stats_patients_with_growth = set()
    stats_max_relevant_tx = 0 
    stats_max_susc_multiplicity = 1 

    for idx, row in data.iterrows():
        patient_id = row["patient id"]

        # --- A. Pre-load ALL Treatments for this Patient ---
        patient_treatments = []
        
        for i in range(tx_steps):
            m_idx = tx_med_idx + (i * tx_step_size)
            d_str_idx = tx_date_str_idx + (i * tx_step_size)
            d_hrs_idx = tx_date_hrs_idx + (i * tx_step_size)
            
            if m_idx >= len(row): break
            
            raw_med = row.iloc[m_idx]
            if pd.notna(raw_med) and str(raw_med).strip() not in ["", "nan", "None", "0"]:
                
                try:
                    hrs_val = float(row.iloc[d_hrs_idx])
                except (ValueError, TypeError):
                    hrs_val = None 
                
                date_str = str(row.iloc[d_str_idx]).strip()
                
                # Store strictly raw data + numeric time for filtering
                patient_treatments.append({
                    "raw_name": str(raw_med).strip(),
                    "date_str": date_str,
                    "hours_val": hrs_val,
                    "hours_str": str(row.iloc[d_hrs_idx]) 
                })

        # --- B. Build Susceptibility Map ---
        # Map: Organism -> { Raw_Lab_Abx_Name : [Result1, Result2...] }
        susc_map = defaultdict(lambda: defaultdict(list))
        for i in range(susc_steps):
            o_idx = susc_org_idx + (i * susc_step_size)
            a_idx = susc_abx_idx + (i * susc_step_size)
            r_idx = susc_res_idx + (i * susc_step_size)
            if r_idx >= len(row): break
            
            s_org = row.iloc[o_idx]
            raw_abx = row.iloc[a_idx]
            s_res = row.iloc[r_idx]
            
            if pd.notna(s_org) and pd.notna(raw_abx) and pd.notna(s_res):
                s_org_str = str(s_org).strip().upper()
                raw_abx_str = str(raw_abx).strip().upper() # Uppercase for consistency
                s_res_str = str(s_res).strip().upper()

                if s_org_str and raw_abx_str and s_res_str not in ["", "NAN", "NONE"]:
                    # No sanitization, just raw lab name
                    susc_map[s_org_str][raw_abx_str].append(s_res_str)

        # --- C. Iterate Growths ---
        patient_had_growth = False
        
        for i in range(growth_steps):
            g_name_i = growth_name_idx + (i * growth_step_size)
            g_date_i = growth_date_idx + (i * growth_step_size)
            if g_name_i >= len(row): break
            
            growth_name = row.iloc[g_name_i]
            growth_days_val = row.iloc[g_date_i] 
            
            if pd.notna(growth_name) and str(growth_name).strip() not in ["", "nan", "None", "0"]:
                growth_name_str = str(growth_name).strip()
                
                # Calculate Growth Time in Hours
                growth_hours = None
                try:
                    growth_hours = float(growth_days_val) * 24.0
                except (ValueError, TypeError):
                    growth_hours = None 
                
                patient_had_growth = True
                
                # --- D. EXPANDED TREATMENT COLUMNS ---
                filtered_txs = []
                
                # Base Row
                new_row = {col: row[col] for col in base_cols}
                new_row['Target_Growth_Name'] = growth_name_str
                new_row['Target_Growth_Date_Days'] = str(growth_days_val)
                
                if growth_hours is not None:
                    window_start = growth_hours - WINDOW_PRE
                    window_end = growth_hours + WINDOW_POST
                    
                    for tx in patient_treatments:
                        if tx["hours_val"] is not None:
                            if window_start <= tx["hours_val"] <= window_end:
                                filtered_txs.append(tx)

                    # Add Dynamic Columns (Tx_1, Tx_2...)
                    count_relevant = len(filtered_txs)
                    if count_relevant > stats_max_relevant_tx:
                        stats_max_relevant_tx = count_relevant
                    
                    for k, tx in enumerate(filtered_txs):
                        prefix = f"Relevant_Tx_{k+1}"
                        new_row[f"{prefix}_Name"] = tx["raw_name"]
                        new_row[f"{prefix}_Date"] = tx["date_str"]
                        new_row[f"{prefix}_Hours"] = tx["hours_str"]

                # --- E. Susceptibility Columns ---
                my_susc_dict = susc_map.get(growth_name_str.upper(), {})
                
                for lab_abx_name, results_list in my_susc_dict.items():
                    if len(results_list) > stats_max_susc_multiplicity:
                        stats_max_susc_multiplicity = len(results_list)
                    
                    # Result Columns Only
                    new_row[f"{lab_abx_name}_Res"] = results_list[0]
                    for k in range(1, len(results_list)):
                        new_row[f"{lab_abx_name}_Res_test{k+1}"] = results_list[k]
                
                new_rows.append(new_row)
        
        if patient_had_growth:
            stats_patients_with_growth.add(patient_id)

    # --- F. Finalize & Sort Columns ---
    output_df = pd.DataFrame(new_rows)
    
    num_growths = len(output_df)
    num_patients = len(stats_patients_with_growth)
    
    if not output_df.empty:
        cols = list(output_df.columns)
        
        ordered_cols = []
        if 'Target_Growth_Name' in cols: ordered_cols.append('Target_Growth_Name')
        if 'Target_Growth_Date_Days' in cols: ordered_cols.append('Target_Growth_Date_Days')
        
        # Sort Tx Columns: Tx_1, Tx_2...
        for k in range(1, stats_max_relevant_tx + 1):
            prefix = f"Relevant_Tx_{k}"
            batch = [f"{prefix}_Name", f"{prefix}_Date", f"{prefix}_Hours"]
            for c in batch:
                if c in cols: ordered_cols.append(c)
        
        remaining = [c for c in cols if c not in ordered_cols]
        if "patient id" in remaining:
            remaining.remove("patient id")
            ordered_cols.insert(0, "patient id")
            
        final_order = ordered_cols + remaining
        output_df = output_df.reindex(columns=final_order)

    print("="*50)
    print("      SMART ANALYSIS STATISTICS (V9 - CLEAN)      ")
    print(f"      Filter Window: -{WINDOW_PRE}h to +{WINDOW_POST}h")
    print("="*50)
    print(f"Total Growth Rows: {num_growths}")
    print(f"Unique Patients with Growth: {num_patients}")
    print(f"Max Relevant Treatments Found for a single growth: {stats_max_relevant_tx}")
    print(f"Max Susceptibility Tests for single Abx: {stats_max_susc_multiplicity}")
    print("="*50)

    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Smart Growth-Centric dataset saved to {output_file}")
    return output_df


def create_growth_centric_dataset_raw(data, output_file="growth_centric_RAW_for_review.csv"):
    """
    RAW Version:
    - Splits patient into multiple rows (one per growth).
    - Cols: [Specific Growth Batch (8 cols)] + [Demographics] + [ALL Abx Cols] + [ALL Susc Cols]
    """
    print("Starting generation of RAW Growth-Centric Dataset...")

    # 1. Define Context Columns
    demographic_cols = [
        "patient id"
        #"patient id", "birth-age_when_documented", "birth-gestational_age", 
        #"maternal_age_above_35", "birth-type_of_labor_onset",
        #"birth-birth_site", "surgery_indication-type_of_surgery",
        #"fever_temperature_numeric_max_37.5-43-numeric_result",
        #"time_from_intrapartum_fever_treatment_hours"
    ]
    valid_demo_cols = [c for c in demographic_cols if c in data.columns]
    
    # Grab ALL original Abx & Susc columns
    abx_cols = [c for c in data.columns if "antibiotics-medication" in c or "antibiotics-date administered" in c]
    susc_cols = [c for c in data.columns if "organisms susceptability" in c]
    
    context_cols = valid_demo_cols + abx_cols + susc_cols

    # 2. Batch Config
    try:
        # We start at 'test type' to capture the full 8-column growth batch
        batch_start_idx = column_name_to_index(data, "cultures-test type_1")
        # 'Organism detected' is usually +2, used for validation
        org_check_idx = column_name_to_index(data, "cultures-organism detected_1")
    except KeyError:
        print("Error: Could not find culture columns.")
        return

    growth_steps = 61
    growth_step_size = 8
    
    # 3. Create Clean Headers for the 8-col batch
    batch_headers_clean = []
    for i in range(8):
        # Grab original name (e.g., "cultures-test type_1")
        col_name = data.columns[batch_start_idx + i]
        # Clean: "Target_Growth_test type"
        clean_name = col_name.replace("_1", "").replace("cultures-", "Target_Growth_")
        batch_headers_clean.append(clean_name)

    new_rows = []

    for idx, row in data.iterrows():
        for i in range(growth_steps):
            curr_batch_start = batch_start_idx + (i * growth_step_size)
            curr_org_check = org_check_idx + (i * growth_step_size)
            
            if curr_batch_start + 8 > len(row): break

            # Valid Growth?
            growth_name = str(row.iloc[curr_org_check]).strip()
            if growth_name and growth_name not in ["", "nan", "None", "0"]:
                
                new_row = {}
                
                # A. The 8 Specific Growth Columns
                for offset in range(8):
                    val = row.iloc[curr_batch_start + offset]
                    new_row[batch_headers_clean[offset]] = val

                # B. The Context
                for col in context_cols:
                    new_row[col] = row[col]
                
                new_rows.append(new_row)

    output_df = pd.DataFrame(new_rows)
    
    # Reorder to put Target cols first
    if not output_df.empty:
        cols = list(output_df.columns)
        for h in reversed(batch_headers_clean):
            if h in cols: cols.insert(0, cols.pop(cols.index(h)))
        if "patient id" in cols:
            cols.insert(0, cols.pop(cols.index("patient id")))
        output_df = output_df[cols]
        
        output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"RAW Growth-Centric dataset saved to {output_file} ({len(output_df)} rows).")

    return output_df


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
    "ANAEROBIC GRAM POSITIVE ROD": "Anaerobes",
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
    "VEILLONELLA SPECIES": "Vaginal Flora",
    "DIALISTER MICRAEROPHILUS": "Anaerobes",
    "ROSEMONAS MUCOSA": "Other Gram Negatives",
    "BACILLUS SIMPLEX": "Contaminants (CONS etc.)"    
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
    
    print(f"Study starting with {len(over_38_data)} deliveries with postpartum fever - {len(unique_patients)} patients")
    
    over_threshold_data = remove_rows_below_threshold(over_38_data, 'birth-gestational age', 24.0)
    print(len(over_38_data)-len(over_threshold_data), " rows below gestational age 24 removed")

    no_termination_data = remove_rows_if_contains(over_threshold_data, 'birth-type of labor onset', ['misoprostol', 'termination of pregnancy','IUFD'])
    print(len(over_threshold_data)-len(no_termination_data), " rows with misoprostol/termination removed")

    data = remove_rows_above_threshold(no_termination_data, 'birth-fetus count', 1)
    print(len(no_termination_data)-len(data), " rows with fetus count above 1 removed")                                                                               

    ## Cultures taken yes/no Follow by Positive yes/no
    data = containswords_andor_containswords_result_exists(data, 'cultures-test type_1', ['דם'], 'OR','cultures-specimen material_1', ['דם'], 8, 61, 'blood_culture_taken')
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
        "5": ["Prev. C.S. - Patient`s Request","Previous Uterine Scar", "S/P Myomectomy"],
        "6": ["Macrosomia"],
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
    #update_column_with_values(data, 'surgery before delivery-procedure_2', words_dict_13, default_value="Other", empty_value="")
                                                                                                                                
    data = update_column_with_values_batch(data, 'surgery after delivery-procedure', words_dict_13, default_value="Other", empty_value="0", batch=4)
                                                                                                                                
    #update_column_with_values(data, 'surgery after delivery-procedure_2', words_dict_13, default_value="Other", empty_value="")

    
    
    ## Remove negative values from 
    #cleared = clear_negative_values(data, '')
    #print(f"{cleared} negative \"third stage length\" values removed")



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
    #premature labor categories - Check if numeric values in column 'gestational age' is in-range or out-of-range for prematurity categories, and add results in a new column '___'
    data = cutoff_range_numeric(data, 'birth-gestational age', 'Late_premature_labor_(34≤GA<37)', 34, 36.6, in_range=1, out_of_range=0, empty_value=None)
    data = cutoff_range_numeric(data, 'birth-gestational age', 'Moderate_early_premature_labor_(32≤GA<34)', 32, 33.6, in_range=1, out_of_range=0, empty_value=None)
    data = cutoff_range_numeric(data, 'birth-gestational age', 'Very_early_premature_labor_(28≤GA<32)', 28, 31.6, in_range=1, out_of_range=0, empty_value=None)
    data = cutoff_range_numeric(data, 'birth-gestational age', 'Extremely_early_premature_labor_(GA<28)', 24, 27.6, in_range=1, out_of_range=0, empty_value=None)
    
    
    #fever before delivery (above 38) - yes/no
    data = cutoff_number(data, 'fever temperature numeric_max 37.5-43-numeric result', 'fever_before_delivery', 38, above=1, below=0, empty_value='')                                                                                                                                     
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

    #data = is_empty(data, 'baby_2_transfer-department', 'baby_2_NICU_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_sga-diagnosis', 'baby_2_SGA_yes_or_no', value_empty=0, value_not_empty=1)     
    #data = is_empty(data, 'baby_2_lga-diagnosis', 'baby_2_LGA_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_meconium aspiration-diagnosis', 'baby_2_meconium aspiration_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_meconium -diagnosis', 'baby_2_meconium_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_hypoglycemia-diagnosis', 'baby_2_hypoglycemia_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_jaundice-diagnosis', 'baby_2_jaundice_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_rds-diagnosis', 'baby_2_RDS_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_ivh-diagnosis', 'baby_2_IVH_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_periventricular leukomalacia -diagnosis', 'baby_2_PVL_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_bronchopulmonary dysplasia -diagnosis', 'baby_2_BPD_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_necrotizing enterocolitis -diagnosis', 'baby_2_NEC_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_seizures-diagnosis', 'baby_2_seizures_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_hypoxic ischemic encephalopathy -diagnosis', 'baby_2_HIE_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_mechanical ventilation-diagnosis', 'baby_2_mechanical ventilation_yes_or_no', value_empty=0, value_not_empty=1)
    #data = is_empty(data, 'baby_2_sepsis-diagnosis', 'baby_2_sepsis_yes_or_no', value_empty=0, value_not_empty=1)
    
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


    data = count_unique_growths_by_dfr(
        data=data,
        dfr_first_col="cultures-collection date-days from reference_1",
        growth_first_col="cultures-organism detected_1",
        num_steps=61,
        step_size=8,
        organism_dict=organism_dict,
        map_column_name="organism_counts_map",
        prefix_for_columns="unique_"
    )

    data = evaluate_appropriate_antibiotic_treatment(
        data,
        abx_med_col="antibiotics-medication_1",
        abx_time_col="antibiotics-date administered-hours from reference_1",
        abx_step_size=3,
        abx_num_steps=108,
        growth_org_col="cultures-organism detected_1",
        growth_time_col="cultures-collection date-days from reference_1",
        growth_step_size=8,
        growth_num_steps=61,
        susc_abx_col="organisms susceptability-antibiotic_1",
        susc_micro_col="organisms susceptability-microorganism_1",
        susc_interp_col="organisms susceptability-susceptibility interpretation_1",
        susc_step_size=5,
        susc_num_steps=65,
        timeframe_before_hours=10,
        timeframe_after_hours=10,
        raw_output_col="abx_growth_susceptibility_raw",
        per_growth_output_col="abx_growth_susceptibility_by_growth",
        overall_output_col="abx_appropriate_treatment_overall"
    )

    # --- INSERT THIS BLOCK ---
    
    # 1. Generate Smart Analysis V9
    create_growth_centric_dataset_smart_v9(data, output_file="output_smart_growth_analysis_v9.csv")
    # 2. Generate Raw Review File (Full 8-col growth batch + Full Context)
    create_growth_centric_dataset_raw(data, output_file="output_raw_growth_review.csv")
    
    # -------------------------

    # Remove specified columns, including single columns and ranges
    data = remove_columns(data, [
        'reference occurrence number',
        'date of birth~date of death - days from delivery',
        'birth-date of first documentation - birth occurence',
        'hospitalization delivery-hospital admission date',
        'hospitalization delivery-hospital discharge date',
        'newborn sheet-hospital admission date_1',
        'newborn sheet-hospital discharge date_1',
        'newborn sheet-apgar 1_2~newborn sheet-child internal patient id_2',
        'newborn sheet-apgar 1_3~newborn sheet-child internal patient id_3',
        'ph_arterial-numeric result_1~ph_arterial-lab test copy(1)_5',
        'bmi-date of measurement-days from reference',
        'second and third stage timeline-time of full dilation',
        'gbs status-gbs in urine',
        'gbs status-gbs vagina',
        'fever temperature numeric_max 37.5-43-date of measurement',
        'onset of fever 38 until delivery-date of measurement-hours from reference',
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
        'baby_2_reference occurrence number~baby_3_apgar -patient id of newborn',
        "birth-fetus count",
        #'baby_2_NICU_yes_or_no~baby_2_mechanical ventilation_yes_or_no',
        #'baby_3_NICU_yes_or_no~baby_3_mechanical ventilation_yes_or_no'

        ])

    data = add_row_index_column(data, col_name="Index")
    data = replace_column_spaces(data)
    data = reorder_columns(data, ["Index",  "patient_id",   "birth-age_when_documented",    "maternal_age_above_35",    "birth-birth_number",   "nulliparous_yesno",    "birth-pregnancy_number",   "birth-type_of_labor_onset",    "birth-gestational_age",    "Extremely_early_premature_labor_(GA<28)",    "Very_early_premature_labor_(28≤GA<32)",    "Moderate_early_premature_labor_(32≤GA<34)",    "Late_premature_labor_(34≤GA<37)",    "birth-birth_site",    "hospitalization_delivery-hospital_length_of_stay",    "Hospital_length_of_stay_above_3d",    "obstetric_formula-number_of_cesarean_sections_(cs)",   "obstetric_formula-number_of_vaginal_births_after_cesarean_sections_(vbac)",    "Previous_CS_yes_no",   "Previous_VBAC_yes_no","pregnancy_conceive-pregnancy_type", "newborn_sheet-apgar_1_1",  "newborn_sheet-apgar_5_1",  "Apgar_1m_below_7", "Apgar_5m_below_7", "newborn_sheet-weight_1",   "newborn_sheet-died_at_pregnancy/birth_1",  "newborn_sheet-gender_1",   "newborn_sheet-sent_to_intensive_care_1",   "newborn_sheet-delivery_type_1",    "newborn_sheet-child_internal_patient_id_1",    "bmi-numeric_result",   "BMI_above_30", "rom_description-amniotic_fluid_color", "rom_description-date_of_membranes_rupture-hours_from_reference",   "duration_of_rom_over_18h", "rom_description-membranes_rupture_type",   "fever_temperature_numeric_max_37.5-43-date_of_measurement-hours_from_reference",   "fever_temperature_numeric_max_37.5-43-numeric_result", "wbc_max-numeric_result",   "crp_max-numeric_result",   "transfers-department", "transfers-department_length_of_stay",  "readmission-admitting_department", "neuraxial_analgesia-anesthesia_type",  "surgery_indication-main_indication",   "surgery_indication-secondary_indication",  "length_of_stay_delivery_room_calculated",  "second_stage_length_calculated",   "duration_of_2nd_stage_over_4h","blood_culture_organisms",  "blood_culture_organisms_category", "Blood_culture_Type_of_growth", "Organisms_Contaminants_yes_or_no", "Organisms_Non_hemolytic_Strep_yes_or_no",  "Organisms_Enterobacterales_yes_or_no", "Organisms_GBS_yes_or_no",  "Organisms_Anaerobes_yes_or_no",    "Organisms_Other_Gram_Negatives_yes_or_no", "Organisms_Vaginal_Flora_yes_or_no",    "Organisms_Staph_Aureus_yes_or_no", "Organisms_Listeria_yes_or_no", "Organisms_Other_yes_or_no",    "Antibiotics_given_Ampicillin", "Antibiotics_given_Augmentin",  "Antibiotics_given_Carbapenem", "Antibiotics_given_Ceftriaxone",    "Antibiotics_given_Clindamycin",    "Antibiotics_given_Gentamycin", "Antibiotics_given_Metronidazole",  "Antibiotics_given_Penicillin", "Antibiotics_given_Tazocin",    "Antibiotics_given_Other",  "GBS_Result",   "Hysterectomy_done_yes_or_no",  "death_at_delivery_yes_no",   "maternal_pregestational_diabetes_yes_or_no", "maternal_gestational_diabetes_yes_or_no",  "maternal_pregestational_hypertension_yes_or_no",   "maternal_gestational_hypertension_yes_or_no",  "maternal_hellp_syndrome_yes_or_no", "maternal_pph_yes_or_no",  "blood_products_given_yes_or_no",   "Surgery_before_delivery",  "Surgery_after_delivery",   "ct_done_yes_or_no",    "drainage_done_yes_or_no",  "pH_Arteiral_Result",   "pH_Arteiral_below_7.1",    "concat_antibiotics_given", "baby_1_transfer-department_discharge_date-days_from_reference",    "baby_1_NICU_yes_or_no",    "baby_1_SGA_yes_or_no", "baby_1_LGA_yes_or_no", "baby_1_meconium_aspiration_yes_or_no", "baby_1_meconium_yes_or_no",    "baby_1_hypoglycemia_yes_or_no",    "baby_1_jaundice_yes_or_no",    "baby_1_RDS_yes_or_no", "baby_1_IVH_yes_or_no", "baby_1_PVL_yes_or_no", "baby_1_BPD_yes_or_no", "baby_1_NEC_yes_or_no", "baby_1_seizures_yes_or_no",    "baby_1_HIE_yes_or_no", "baby_1_sepsis_yes_or_no",  "baby_1_mechanical_ventilation_yes_or_no",  "neonetal_death_yes_no",    "time_from_intrapartum_fever_treatment_hours",  "time_from_intrapartum_fever_treatment_hours_abx",  "GBS_prophylactic_treatment_yes/no"])
    print(f"done reordering columns")



    save_data(data, output_filepath)
    #split_and_save_csv(data, 'fever temperature numeric_max 37.5-43-numeric result', 'output.csv', 'output_under_38.csv', 'output_38_or_above.csv', encoding='utf-8')

if __name__ == "__main__":
    main()