import pandas as pd
import re
import numpy as np
import os
import ast


def date_convert(df):
    # Convert 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'], utc=True)

    # Extract the desired format 'day month year' from the datetime format
    df['formatted_date'] = df['date'].dt.strftime('%d %b %Y')

    # Add a column for Unix timestamp
    df['unix_time'] = df['date'].apply(lambda x: int(x.timestamp()))
    
    return df

def filter_date(df, target_date):
    mask = df['date'] >= target_date
    return df[mask]

def normalize_sender_name(name):
    name = str(name)
    match = re.search(r'<([^>]+)>', name)
    if match:
        sender_name = match.group(1)
        sender_name = re.sub(r'\b[a-z]\b', '', sender_name, flags=re.IGNORECASE).strip()
        return sender_name.title()
    return ""


def replace_unicode_escapes(match):
    return chr(int(match.group(1), 16))

def normalize_text(text):
    try:
        normalized_text = text.strip().replace('\xa0', ' ')
        normalized_text = re.sub(r'\\u([0-9a-fA-F]{4})', replace_unicode_escapes, normalized_text)
        normalized_text = re.sub(r'(\n\s*)+', '\n', normalized_text)
        return normalized_text
    except:
        return text
    
    
def email_text_divider(text, primary_dividers, secondary_dividers=['From:', '\nGet Outlook', 'wrote:', 'Sent from my']):
    """
    Divides the email text at the first occurrence of any primary dividers or any of the secondary dividers using regex.

    :param text: The email text.
    :param primary_dividers: A list or a single primary divider.
    :param secondary_dividers: A list of secondary dividers.
    :return: The text before the first occurrence of any divider.
    """
    if not isinstance(text, str):
        return text

    try:
        # Ensure primary_dividers is a list
        if isinstance(primary_dividers, str):
            primary_dividers = [primary_dividers]
        elif not isinstance(primary_dividers, list):
            primary_dividers = []

        # Combine all dividers into a single list
        all_dividers = primary_dividers + secondary_dividers

        # Create a regex pattern to match any of the dividers
        dividers_pattern = '|'.join(re.escape(div) for div in all_dividers if div)
        
        match = re.search(dividers_pattern, text)
        if match:
            return text[:match.start()]
        else:
            return text
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    
def get_signature_dividers(sender_id_list, personnel_df):
    if isinstance(sender_id_list, str):
        try:
            sender_id_list = ast.literal_eval(sender_id_list)
        except ValueError:
            print("Invalid sender_id format.")
            return []

    if not sender_id_list or not isinstance(sender_id_list, list) or sender_id_list[0] == -1:
        return []

    try:
        sender_id = sender_id_list[0]
        divider_entry = personnel_df[personnel_df['personnel_id'] == sender_id]['signature_divider'].iloc[0]

        # Convert the string representation of a list into an actual list
        if isinstance(divider_entry, str):
            try:
                divider_entry = ast.literal_eval(divider_entry)
            except ValueError:
                # In case the conversion fails, treat it as a single string divider
                divider_entry = [divider_entry]
        
        if not isinstance(divider_entry, list):
            divider_entry = [divider_entry]  # Ensure it's always a list

        # Filter out any non-string dividers
        return [div for div in divider_entry if isinstance(div, str)]
    except IndexError:
        print("Sender ID not found in personnel_df.")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

def cleaning_email(df, personnel_df):
    df['divided_text'] = df.apply(lambda row: email_text_divider(row['raw_text'], 
                              get_signature_dividers(row['sender_id'], personnel_df)), axis=1)    
    df['cleaned_text'] = df['divided_text'].apply(normalize_text)
    unique_id = df['external_email_id'].unique()
    email_to_id_mapping = {email: f'eml_{index+1}' for index, email in enumerate(unique_id)}
    df['internal_email_id'] = df['external_email_id'].map(email_to_id_mapping)
    # last_column = df.pop(df.columns[-1])
    # df.insert(0, last_column.name, last_column)
    return df

def extract_and_match_personnel_ids(df, personnel_df, input_column):
    # Extract email addresses enclosed in angle brackets from the input_column
    df[input_column].fillna('', inplace=True)
    df['extracted_email'] = df[input_column].apply(lambda x: re.findall(r'<(.*?)>', x))
    
    # Initialize a list to store personnel_id values
    personnel_id_list = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        extracted_email_list = row['extracted_email']
        matching_personnel_ids = set()

        # Iterate through the extracted emails for the current row
        for extracted_email in extracted_email_list:
            condition = 0
            # Check if the extracted email exists in the 'personnel_df' dataframe
            for personnel_index, personnel_row in personnel_df.iterrows():
                if extracted_email in personnel_row['email']:
                    matching_personnel_ids.add(personnel_row['personnel_id'])
                    condition = 1
            if condition == 0:
                if all(substring in extracted_email for substring in ['illinois', 'edu']):
                    matching_personnel_ids.add(-1)
                else:
                    matching_personnel_ids.add(-2)

        # Append matching_personnel_ids to personnel_id_list for the current row, or add NaN if no match was found
        personnel_id_list.append(list(matching_personnel_ids))
        # if matching_personnel_ids:
        #     personnel_id_list.append(matching_personnel_ids)
        # else:
        #     personnel_id_list.append(matching_personnel_ids)

    # Add the 'matching_personnel_ids' as a new column right after the input column
    if f'{input_column}_id' in df.columns:
        df[f'{input_column}_id'] = personnel_id_list
    else:
        df.insert(loc=df.columns.get_loc(input_column) + 1, column=f'{input_column}_id', value=personnel_id_list)
    
    # Drop the intermediate 'extracted_email' column
    df.drop(columns='extracted_email', inplace=True)

    return df

def main():
    current_directory = os.getcwd()
    parent_directory = os.path.abspath(os.path.join(current_directory, ".."))

    data_path = '/projects/bcad/coin/data'
    per_path = '/projects/bcad/coin/personnel/personnel.csv'
    save_path = 'projects/bcad/coin/data/whole_cleaned.csv'

    df_eml = pd.read_csv(data_path)

    personnel_df = pd.read_csv(per_path)

    df_eml = extract_and_match_personnel_ids(df_eml, personnel_df, 'sender')
    df_eml = extract_and_match_personnel_ids(df_eml, personnel_df, 'recipients')
    df_eml = extract_and_match_personnel_ids(df_eml, personnel_df, 'cc')

    df_cleaned = cleaning_email(df_eml, personnel_df)

    df_cleaned.to_csv(save_path, index=False)

if __name__: # main function
    main()
    pass