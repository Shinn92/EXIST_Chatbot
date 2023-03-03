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


openai.api_key = st.secrets["api_key"]


start_time=time.time()

########### This helps takes care of removing metadata
search_string = "---" 
metadata_counter = 0
############




messages = [
    {"role": "system", "content": ""},
]

#Erstelle einen Datframe mit Inhalten und den dazugehörigen Embeddings
df_try =pd.read_csv('df_chatbot_exist_v3.csv')
all_embeddings = np.load('embeddings.npy', allow_pickle=True)
df_try['ada_v2_embedding'] = all_embeddings



def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def search_docs(df, user_query, top_n=3, to_print=True):
    embedding = get_embedding(
        user_query,
        model="text-embedding-ada-002"
    )
    #Erstelle eine Kopie des Datframes  
    df_question = df.copy()
    
    #Füge eine weitere Spalte hinzu mit Similarity-Score
    df_question["similarities"] = df.ada_v2_embedding.apply(lambda x: cosine_similarity(x, embedding))
    
    #Sortiere den Dataframe basierend auf dem Similarity-Score und zeige die obersten N Einträge
    res = (
        df_question.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    #Greife die obersten N Einträge ab
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
            #Greife den Eintrag ab, der am meisten Änhlichkeit mit der Frage hat
            res = search_docs(df_try, ai_question, top_n=1)
            #Greife den Inhalt des Eintrages ab
            context= res.CONTENT.values
            #Kombiniere den Prompt mit Baisisprompt, dem Inhalt und der Frage
            combined_prompt = initial_prompt + str(context) + "Q: " + ai_question
            
            #API-Abfrage
            messages.append(
            {"role": "user", "content": combined_prompt},
        )        
            chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
            
            #Greife Inhalt des Resultat der API Abfrage ab
            ai_response = chat.choices[0].message.content
            #st.write('')
            st.write(ai_response)
           # st.write('______________________________________________________________________________________')
            #st.write('')



