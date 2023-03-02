#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import io
import time
import pandas as pd
import openai
import numpy as np
import streamlit as st
import requests

from openai.embeddings_utils import cosine_similarity
from num2words import num2words


OPENAI_API_KEY = st.secrets["api_key"]

csv_url = 'https://github.com/Shinn92/EXIST_Chatbot/blob/main/Files/df_chatbot_exist_v2.csv'
npy_url = 'https://github.com/Shinn92/EXIST_Chatbot/blob/main/Files/embeddings.npy'
start_time=time.time()

########### This helps takes care of removing metadata
search_string = "---" 
metadata_counter = 0
############
req_npy = requests.get(npy_url)
req_csv = requests.get(csv_url)
#csv_file = "temporaryCSV.csv"
npy_file = "temporaryNpy.npy"

result = requests.get(csv_url)
content = result.content.decode("utf-8")


#with open(csv_file, 'wb') as f:
    #f.write(req_csv.content)
    
#with open(npy_file, 'wb') as f:
    #f.write(req_npy.content)

messages = [
    {"role": "system", "content": ""},
]

#df_try = pd.read_csv('df_chatbot_exist_v2.csv', encoding='utf-8')
#df_try = pd.read_csv(io.StringIO(content))
df_try =pd.read_csv('df_chatbot_exist_v3.csv')
#df_try = pd.read_csv(csv_file, encoding='utf-8')
all_embeddings = np.load('embeddings.npy', allow_pickle=True)
#all_embeddings = np.load(npy_file, allow_pickle=True)
df_try['ada_v2_embedding'] = all_embeddings


df_similarities = df_try.copy()

def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def search_docs(df, user_query, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        model="text-embedding-ada-002"
    )
    
    df_question = df_similarities.copy()
    df_question["similarities"] = df_try.ada_v2_embedding.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df_question.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    return res


    
if __name__== '__main__':
    #while True:
        keyInt = 0
        initial_prompt ="Du bist ein Start-Up Guide für das EXIST Programm. Du bist hilfreich, clever und freundlich. Antworte ausschließlich in deutscher Sprache. Sei prägnant in deinen Antworten. Nutze den folgenden Text für deine Antwort:"
        
        ai_question = st.text_input('Wie kann ich dir helfen???', key=str(keyInt))
        # Create a button to submit the inputs
        submit = st.button('Abschicken')
        # Check if the button is pressed
        if submit:
            keyInt = keyInt + 1
            res = search_docs(df_try, ai_question, top_n=1)
            context= res.CONTENT.values
            combined_prompt = initial_prompt + str(context) + "Q: " + ai_question
            messages.append(
            {"role": "user", "content": combined_prompt},
        )        
            chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )

            ai_response = chat.choices[0].message.content
            #st.write('')
            st.write(ai_response)
           # st.write('______________________________________________________________________________________')
            #st.write('')



