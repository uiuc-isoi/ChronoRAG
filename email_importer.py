import pandas as pd
import os
import email
import hashlib

import re
import numpy as np
from bs4 import BeautifulSoup
import glob
from tqdm import tqdm
import warnings
from bs4 import GuessedAtParserWarning
warnings.filterwarnings('ignore', category=GuessedAtParserWarning)

hash_db_path = os.path.join("..", "..", "data", "email_data_imported", "attachment", "hash_database.csv")
saved_files_count = 0
skipped_files_count = 0

def load_eml_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors="ignore") as eml_file:
        return email.message_from_file(eml_file)
    
def collect_attachment_filenames(part, internal_email_id):
    """
    Generate the new filename for an attachment based on the internal email ID.
    """
    filename = part.get_filename()
    new_filename = f"{internal_email_id}_{filename}" if filename else None
    return new_filename

def init_hash_database():
    """Initialize or load the hash database."""
    if os.path.exists(hash_db_path):
        return pd.read_csv(hash_db_path)
    else:
        return pd.DataFrame(columns=['hash_value', 'filename', 'extension'])

def file_hash(content):
    """Generate SHA-256 hash of file content."""
    hasher = hashlib.sha256()
    hasher.update(content)
    return hasher.hexdigest()

def check_and_save_unique(email_part, filepath, hash_database):
    """Check if the attachment is unique and save it."""
    # Assuming 'file_hash' and 'init_hash_database' are defined as before
    global saved_files_count, skipped_files_count
    
    temp_content = email_part.get_payload(decode=True)
    hash_value = file_hash(temp_content)
    
    file_name, file_extension = os.path.splitext(os.path.basename(filepath))
    
    if hash_value not in hash_database['hash_value'].values:
        try:
            # Attempt to save the file
            with open(filepath, "wb") as f:
                f.write(temp_content)
            
            # If save succeeds, update the DataFrame
            file_name, file_extension = os.path.splitext(os.path.basename(filepath))
            new_row_df = pd.DataFrame({
                'hash_value': [hash_value], 
                'filename': [file_name], 
                'extension': [file_extension]
            })
            hash_database = pd.concat([hash_database, new_row_df], ignore_index=True)
            
            saved_files_count += 1
        except:
            pass
    else:
        skipped_files_count += 1
    
    return hash_database

def extract_text_and_metadata_from_email(eml_message, internal_email_id, hash_database, attachment_dir="attachment"):
    text = ''
    attachments = []

    sender = eml_message.get('From', '')
    recipients = eml_message.get('To', '')
    cc = eml_message.get('Cc', '')
    subject = eml_message.get('Subject', '')
    timestamp = eml_message.get('Date', '')
    external_email_id = eml_message.get('Message-ID', '')
    in_reply_to = eml_message.get('In-Reply-To', '')
    references = eml_message.get('References', '')
    reply_to = eml_message.get('Reply-To', '')
    bcc = eml_message.get('Bcc', '')
    received = eml_message.get_all('Received', '')

    paragraph_list = []
    text = ''
    for part in eml_message.walk():
        content_type = part.get_content_type()
        content_disposition = str(part.get("Content-Disposition"))
        
        if content_disposition and "attachment" in content_disposition:        
            try:
                attachment_filename = collect_attachment_filenames(part, internal_email_id)
                if attachment_filename:  # Check if there's a filename
                    # Extract file extension and prepare directory path for it
                    file_extension = os.path.splitext(attachment_filename)[1].lstrip('.').lower()
                    if file_extension:  # Ensure there is an extension
                        extension_dir = os.path.join(attachment_dir, file_extension)
                        # Ensure the extension directory exists
                        if not os.path.exists(extension_dir):
                            os.makedirs(extension_dir)
                        filepath = os.path.join(extension_dir, attachment_filename)
                    else:
                        # Default directory for files without an extension
                        default_dir = os.path.join(attachment_dir, "no_extension")
                        if not os.path.exists(default_dir):
                            os.makedirs(default_dir)
                        filepath = os.path.join(default_dir, attachment_filename)
                    
                    hash_database = check_and_save_unique(part, filepath, hash_database)

            except Exception as e:
                try:
                    log_message = f"Error: {e}\nInternal Email ID: {internal_email_id}\n"
                    log_filepath = os.path.join(attachment_dir, "attachment_log.txt")
                    with open(log_filepath, "a") as log_file:
                        log_file.write(log_message)
                except:
                    pass

        if "attachment" not in content_disposition:
            if content_type == "text/plain":
                text += part.get_payload(decode=True).decode("utf-8", errors="ignore")
                # Split text by one or more newline characters to get paragraphs
                paragraphs = [line.strip() for line in text.split('\n\n') if line.strip()]
                # Optionally, extend your paragraph list if you are aggregating paragraphs
                paragraph_list.extend(paragraphs)
            elif content_type == "text/html":
                html_text = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                soup = BeautifulSoup(html_text, "html.parser")
                text += soup.get_text()
                # Find all paragraph tags
                paragraphs = soup.find_all('p')

                # Extract content from each paragraph
                paragraph_list = [paragraph.get_text() for paragraph in paragraphs]  # for text content


    return hash_database, text, paragraph_list, attachments, sender, recipients, cc, subject, timestamp, external_email_id, in_reply_to, references, reply_to, bcc, received

def explore_directory(data_path, save_path, hash_database):
    df_list = []
    seen_external_email_ids = set()
    email_counter = 0
    save_interval = 1000
    hash_database = hash_database
    attachment_path = os.path.join("..", "..", "data", "email_data_imported", "attachment", "raw")

    for eml_file_path in tqdm(glob.glob(os.path.join(data_path, "**", "*.eml"), recursive=True)):
        eml_message = load_eml_file(eml_file_path)
        external_email_id = eml_message.get('Message-ID', '')

        if external_email_id in seen_external_email_ids:
            continue
        else:
            seen_external_email_ids.add(external_email_id)
            email_counter += 1

        internal_email_id = f"eml_{email_counter}"
        hash_database, parsed_text, paragraph_list, attachments, sender, recipients, cc, subject, timestamp, message_id, in_reply_to, references, reply_to, bcc, received = extract_text_and_metadata_from_email(eml_message, internal_email_id, hash_database=hash_database, attachment_dir=attachment_path)
        file_name = os.path.basename(eml_file_path)[:-4]
        dirname = os.path.dirname(eml_file_path)

        pattern = r'"(.*?)"'
        match = re.search(pattern, sender)
        extracted_sender = match.group(1) if match else None
        
        df_data = {
            'file_name': [file_name],
            'sender_raw': [sender],
            'sender': [extracted_sender],
            'recipients': [recipients],
            'cc': [cc],
            'subject': [subject],
            'timestamp': [timestamp],
            'external_email_id': [external_email_id],
            'internal_email_address': [internal_email_id],
            'raw_text': [parsed_text],
            'paragraph': [paragraph_list],
            'attachments': [attachments], 
            'in_reply_to': [in_reply_to],
            'references': [references],
            'reply_to': [reply_to],
            'directory': [dirname],
            'bcc': [bcc],
            'received': [received],
        }
        
        df_eml = pd.DataFrame(df_data)
        df_list.append(df_eml)



        if len(df_list) == 100:
            print("Test successful.", flush=True)
            test_df = pd.concat(df_list, ignore_index=True)
            print(test_df['external_email_id'][0], flush=True)
            hash_database.to_csv(hash_db_path, index=False)


        if len(df_list) % save_interval == 0:
            print(f"Processed {len(df_list)} files", flush=True)

            if df_list:
                final_df = pd.concat(df_list, ignore_index=True)
            else:
                final_df = pd.DataFrame()

            hash_database.to_csv(hash_db_path, index=False)
            final_df.to_csv(f'{save_path}/whole_raw.csv', index=False, encoding='utf-8', errors='replace')


    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
    else:
        final_df = pd.DataFrame()
    
    final_df.to_csv(f'{save_path}/whole_raw.csv', index=False, encoding='utf-8', errors='replace')
    hash_database.to_csv(hash_db_path, index=False)

    global saved_files_count, skipped_files_count
    print(f"Saved {saved_files_count} files and skipped {skipped_files_count} files", flush=True)


    return final_df

def main():
    current_directory = os.getcwd()
    parent_directory = os.path.abspath(os.path.join(current_directory, "..", '..'))
    read_path = os.path.join(parent_directory, "data/email_data_raw")
    save_path = os.path.join(parent_directory, "data/email_data_imported")
    hash_database = init_hash_database()
    df_eml = explore_directory(read_path, save_path, hash_database)
    return df_eml


if __name__: # main function
    main()
    pass