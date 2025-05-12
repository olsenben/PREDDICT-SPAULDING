import pandas as pd
import numpy as np
import json
import os

"""
Run this script to processed Form1.csv. Make sure your data is in the correct directory!

...PREDDICT-Spaulding\Database\TBIMSPublic.2024-11-01\Data

KNOWN ISSUES: B3TCOMP,	B3TEF,	B3TEM,	BackCountDigits_i_n,	BackDigitCorrect_i_n, DelayWordRecallCorrect_i_n,	FluencyCorrect_i_n,
DelayWordRecallCorrect_i_n,	FluencyCorrect_i_n, ReasonCorrect_i_n,	WordRecallCorrect_i_n  have some weird outputs that need to be validated.

Some np.nan values are casting to strings in weird ways that will need to be addressed. super weird 

 """


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
        temp_series = temp_series.apply(lambda x: str(float(x)) if pd.notna(x) and str(x).replace('.', '', 1).isdigit() else str(x))
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

#open csv
base_dir = os.path.dirname(__file__)  # directory of this script

csv_path = os.path.join(base_dir, '..', 'Database', 'TBIMSPublic.2024-11-01', 'Data', 'Form1.csv')

df = pd.read_csv(csv_path)

#open data dictionary
json_filepath = os.path.join(base_dir, '..', 'Database', 'processed', 'code_dict.json')

with open(json_filepath, 'r') as file:
    code_dict = json.load(file)

#remove codes from quantitative data, keep data in place
cont_cols = code_dict['numeric_cols']

df_cont = df[cont_cols]

df_cont_clean = process_mixed_variables(df_cont, code_dict, replace_col=True, code_col=True, errors='coerce')

#remove codes from qualitative data, keep data in place
cat_cols = code_dict['categorical_cols']

df_cat = df[cat_cols]

#Theres a lot of codes that tell us data is missing. They dont add much value, but I will keep record in the _code column.
missing_data_codes_dict = {

            '999' : 'Unknown',
            '666' : 'Variable Did Not Exist',
            '777' : 'Refused',
            '888' : 'Not Applicable',
            '99' : 'Unknown',
            '66' : 'Variable Did Not Exist',
            '77' : 'Refused',
            '88' : 'Not Applicable',
            '8888' : 'Not Applicable',
            '9999' : 'Unknown',
            '6666' : 'Variable Did Not Exist',
            '7777' : 'Refused'
}

cat_col_missing_data_dict = {}

for col in cat_cols:
    cat_col_missing_data_dict[col] = missing_data_codes_dict

df_cat_clean = process_mixed_variables(df_cat, cat_col_missing_data_dict, replace_col=True, code_col=True, errors='ignore')

#combine
df_clean = pd.concat([df_cont_clean, df_cat_clean], axis=1)

#data cleaning: this is by no means exhaustive and based on my brief graphical inspection

#remove negative ages
df_clean['AGENoPHI'] = df_clean['AGENoPHI'].mask(df_clean['AGENoPHI'] <= 0.0, np.nan)

#inputing ages as categories. 
bins = [11, 17, 24, 34, 44, 54, 64, 74, 84, 88, float('inf')]
labels = range(10)

df_clean['AgeGroup'] = pd.cut(df_clean['AGENoPHI'].fillna(-1), bins=bins, labels=labels, right=True)

df_clean.loc[df_clean['AGENoPHI'] == 777.0, 'AgeGroup'] = 9

#BackCountTime is basically just over or under 30 seconds with the vast majority over 30 seconds
#create new binary column for BackCountTime_over_30, 1=True, 0=False
df_clean['BackCountTime'].astype(float)
df_clean['BackCountTime_over_30'] = np.where(
    df_clean['BackCountTime'] >= 30, 1,
    np.where(df_clean['BackCountTime'].isna(), np.nan, 0))

df_clean.loc[df_clean['BackCountTime'] >= 30.0, 'BackCountTime_over_30'] = 1

#DeathCause1 and DeathCause2 are categorical and use ICD-10-CM Codes 
#ZipInj is just zip codes
#DeathECode is categorical and has a few random codes.
#all these can be converted to str unless needed for ml purposes
str_cols = ['DeathCause1', 'DeathCause2', 'DeathECode', 'ZipInj']
df_str_cols = df_clean[str_cols]
df_str_cols_clean = process_mixed_variables(df_str_cols, code_dict, replace_col=True, code_col=True, errors='ignore')
df_clean = df_clean.drop(columns=df_str_cols, errors='ignore')
df_clean = pd.concat([df_clean, df_str_cols_clean], axis=1)

for col in str_cols:
    df_clean[col] = df_clean[col].astype(str).str.zfill(5)

#FluencyInt has a few outliers 
df_clean['FluencyInt'] = df_clean['AGENoPHI'].mask((df_clean['FluencyInt'] <= 0.0) | (df_clean['FluencyInt'] > 100), np.nan)

#PTAMethod has an additional code 9, since there are so few I will drop these as an error
df_clean['PTAMethod'] = df_clean['PTAMethod'].mask(df_clean['PTAMethod'] == 9.0, np.nan)


output_path = os.path.join(base_dir, '..', 'Database', 'processed')

filename = os.path.join(output_path, 'form_1_cleaned.csv')

if not os.path.exists(output_path):
    os.makedirs(output_path)

df_clean.to_csv(filename)