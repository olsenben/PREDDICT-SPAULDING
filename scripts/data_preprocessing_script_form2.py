import pandas as pd
import numpy as np
import json
import os

def normalize_code(code):
    """ will convert to format 'string.0' """
    #try to convert to a float directly
    try:
        return str(float(code))
    #if it can't be converted to a float, save it as a string
    except ValueError:
        return str(code)

def process_mixed_variables(df, value_labels_dict, replace_col=False, code_col=False, code_label_col=False,errors="coerece"):
    """
    Cleans coded categorical/numeric variables using a value label dict.
    Returns a cleaned dataframe and a summary.
    """
    df = df.copy()
    original_cols = df.columns
    col_map = {col.lower(): col for col in original_cols}

    new_columns = {}

    for var_raw, codes_dict in value_labels_dict.items():
        var_lower = var_raw.lower()

        if var_lower not in col_map:
            continue

        original_var = col_map[var_lower]

        # Work on a lowercase temporary series for logic
        temp_series = df[original_var].copy()

        # Normalize for mapping
        code_keys = set(normalize_code(k) for k in codes_dict.keys())

        # Convert entries for numeric masking
        temp_series = temp_series.apply(lambda x: str(float(x)) if pd.notna(x) and str(x).replace('.', '', 1).isdigit() else x)
        clean_series = pd.to_numeric(temp_series.mask(temp_series.isin(code_keys)), errors=errors)

        mapped_dict = {str(float(k)) if k.isdigit() else k: v for k, v in codes_dict.items()}
        label_series = temp_series.map(mapped_dict)

        # Output columns using original name
        if replace_col:
            new_columns[original_var] = clean_series
        else:
            new_columns[f'{original_var}_clean'] = clean_series

        if code_col:
            new_columns[f'{original_var}_code'] = temp_series.where(temp_series.isin(code_keys))
        
        if code_label_col:
            new_columns[f'{original_var}_code_label'] = label_series
    

    # Drop old versions of columns being replaced
    df = df.drop(columns=new_columns.keys(), errors='ignore')
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df

def read_json(filename,dir):
    """ load json file"""
    path = os.path.join(dir, filename)

    if os.path.exists(path):
        with open(path, 'r') as f:
           data = json.load(f)
        return data
    
    else:
        print(f"File {filename} not found at {dir}")
        return None
    
def save_csv(df, dir, filename):
    filename = os.path.join(dir, filename)

    if not os.path.exists(dir):
        os.makedirs(dir)
    
    df.to_csv(filename, index=False)
    
def remove_negative_values(df, columns, replace_value=np.nan): 
    """remove negative values from certain columns"""
    for col in columns:
        df.loc[df[col] < 0, col] = replace_value

def bin_data(df, bins=[11, 17, 24, 34, 44, 54, 64, 74, 84, 88, float('inf')], labels=range(10), fill=-1, right=True):
    """bins data"""
    return pd.cut(df.fillna(fill), bins=bins, labels=labels, right=right)


if __name__ == "__main__":

    #open csv
    print("loading data...")
    base_dir = os.path.dirname(__file__)  # directory of this script
    csv_path = os.path.join(base_dir, '..', 'Database', 'TBIMSPublic.2024-11-01', 'Data', 'Form2.csv')
    df_raw = pd.read_csv(csv_path)

    #open data dictionary
    code_dict_dir = os.path.join(base_dir, '..', 'Database', 'processed')
    code_dict = read_json('code_dict_form2.json', code_dict_dir)

    record_cols = code_dict['record_columns']
    cat_cols = code_dict['cat_cols']
    cont_cols = code_dict['cont_cols']

    #process
    print('Processing variables...')
    df_cat = process_mixed_variables(df_raw[cat_cols], code_dict, replace_col=True, code_col=False, errors='ignore')
    df_cont = process_mixed_variables(df_raw[cont_cols], code_dict, replace_col=True, code_col=False, errors='coerce')
    df_records = df_raw[record_cols].copy()
    
    #concat and sort
    print("Cleaning data...")
    df = pd.concat([df_records, df_cat, df_cont], axis=1)
    
    
    #remove negative values
    neg_cols = code_dict['cols_with_neg']
    remove_negative_values(df,neg_cols)

    #bin age group (put original age column back in since 777 was removed (777=89+))
    df['AGENoPHIF'] = df_raw['AGENoPHIF']
    remove_negative_values(df,['AGENoPHIF'])
    df['AgeGroup'] = bin_data(df['AGENoPHIF'])

    #remove "Unknown" from CombinedDRSTypeF
    df['CombinedDRSTypeF'] = df['CombinedDRSTypeF'].replace('Unknown', np.nan)

    df = df.sort_index(axis=1)

    #save output
    print("Saving data...")
    save_dir =  os.path.join(base_dir, '..', 'Database', 'processed')
    save_csv(df,save_dir,'form_2_cleaned.csv' )
    

