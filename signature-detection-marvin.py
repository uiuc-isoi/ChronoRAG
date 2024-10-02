# Python code to separate email signature from content using Marvin AI tool

import pandas as pd
import marvin
from pydantic import BaseModel, Field

import sys
import logging
import os
from dotenv import load_dotenv

# create and configure logger
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%dT%H:%M:%S',
                    format='%(asctime)-15s.%(msecs)03dZ %(levelname)-7s : %(name)s - %(message)s',
                    handlers=[logging.FileHandler("llm.log"), logging.StreamHandler(sys.stdout)])
# create log object with current module name
log = logging.getLogger(__name__)

# Load .env file
load_dotenv()

marvin.settings.openai.api_key = os.getenv('OPENAI_API_KEY')
marvin.settings.openai.chat.completions.model = 'gpt-3.5-turbo'


class EmailParser(BaseModel):
    content: str = Field(..., description="content of email")
    signature: str = Field(..., description="full email signature")

# read cleaned_text column values from test_data.csv file
data = pd.read_csv("./test_data.csv")
parsed_content = []
parsed_signature = []

for text in data['cleaned_text']:
    try:
        result = marvin.cast(text, target=EmailParser)
        parsed_content.append(result.content)
        parsed_signature.append(result.signature)
    except:
        parsed_content.append("NA")
        parsed_signature.append("NA")

data['content'] = parsed_content
data['signature'] = parsed_signature

data.to_csv("results.csv")


