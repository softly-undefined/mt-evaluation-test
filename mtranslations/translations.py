from openai import OpenAI
import pandas as pd
import anthropic
from tqdm import tqdm
import os

# Eric Bennett, 7/29/24
#
# Still need to add google translate functionality! !

USE_AI = True
openai_api_key = "" #paste api key here
anthropic_api_key = "" #paste other api key here
translation_models = ['claude-3-haiku-20240307','claude-3-sonnet-20240229'] #names of the translation models to try
#note googletrans is used for google translate translations


# got all this code from my translation interface: https://github.com/softly-undefined/classical-chinese-tool-v2
class Config:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        

config = Config()

# establish the two clients for use later
config.openai_client = OpenAI(api_key=openai_api_key)
config.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)


def translate(text, aimodel):
    if USE_AI:
        if "gpt" in aimodel.lower(): # Make an OPENAI api call
            return openai_api_call(text, aimodel)
        else: #Make an anthropic api call
            return anthropic_api_call(text, aimodel)
            
    else: 
        return "example translated text "

def openai_api_call(text, aimodel):
    completion = config.openai_client.chat.completions.create(
                model=aimodel,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI model trained to translate Classical Chinese to English, translate the given text to English"
                    },
                    {
                        
                        "role": "user",
                        "content": text,
                    },
                ]
            )
    return completion.choices[0].message.content

def anthropic_api_call(text, aimodel):
    message = config.anthropic_client.messages.create(
            model=aimodel, #"claude-3-opus-20240229"
            max_tokens=1000,
            temperature=0,
            system="Take the input Chinese text and translate it to English without using any new-line characters ('\\n') outputting English text",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
            )
    return message.content[0].text

#
# Interfacing (correctly manipulating the different datasets to translate the right things)
#



df2008 = pd.read_csv("../mt-dataset/cwmt2008_ce_news.tsv", delimiter='\t')
df2009 = pd.read_csv("../mt-dataset/cwmt2009_ce_news.tsv", delimiter='\t')

# df2008.dropna()
# df2009.dropna()

# this part loads in existing translation data to avoid replicating the translations (ex. if already did GPT4 won't redo)
if os.path.exists("translations2008.csv"):
    translations2008 = pd.read_csv("translations2008.csv")
else:
    translations2008 = pd.DataFrame()

if os.path.exists("translations2009.csv"):
    translations2009 = pd.read_csv("translations2009.csv")
else:
    translations2009 = pd.DataFrame()


# translates both datasets using every model, storing data in respective dataframes)
for model in tqdm(translation_models, desc="Translation Models: "):
    data = []
    if  model not in translations2008.columns:
        for _, row in tqdm(df2008.iterrows(), desc="2008 Translations: "):
            text = row['src']
            translated = translate(text, model)
            data.append(translated)
        translations2008[model] = data
    
    data = []
    if model not in translations2009.columns:
        for _, row in tqdm(df2009.iterrows(), desc="2009 Translations: "):
            text = row['src']
            translated = translate(text, model)
            data.append(translated)
        translations2009[model] = data



translations2008.to_csv("translations2008.csv", index=False)
translations2009.to_csv("translations2009.csv", index=False)


