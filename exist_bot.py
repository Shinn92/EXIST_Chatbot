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
#openai.api_key = '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'

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


# Set page config
st.set_page_config(page_title="EXIST-Chatbot", page_icon=":robot_face:", layout="wide")


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.meercommunity.de/wp-content/uploads/2022/09/Hover_bg.png");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 


    

def get_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']



def update_markdown(empty_slot, new_text):
    empty_slot.markdown(f"""
        <style>
            .my-container {{
                background-color: #444076;
                color: white;
                font-size: 24px;
            }}
        </style>
        <div class="my-container">
            {new_text}
        </div>
    """, unsafe_allow_html=True)

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

ai_response = ""
ai_question = ""
    
if __name__== '__main__':
    #while True:
    keyInt = 0
    #input = ""
    # Render page layout
        
    initial_prompt ="Du bist ein Start-Up Guide für das EXIST Programm. Du bist hilfreich, clever und freundlich. Antworte ausschließlich in deutscher Sprache. Sei prägnant in deinen Antworten. Nutze den folgenden Text für deine Antwort:"
    #ai_question = st.text_input('Wie kann ich dir helfen???', key=str(keyInt))
            # Define colors
    bg_color = "#211f39" # marine blue
    text_color = "#ffffff" # white
    
    # Define logos
    logo1 = "MCSC_Logo.png"
    logo2 = "Koop_Innosys_NW.png"
    logoBMWK ="logo_bmwi.png"
    logoHSEL ="logo_hsel.png"
    logoEXIST ="logo_exist.png"
    notice = "hinweis_beta.png"
    
    # Set page background color
    page_bg = f"background-color: {bg_color};"
    st.markdown("<style>body{background-color: Blue;}</style>",unsafe_allow_html=True)
    
    # Create header section
    header_bg = f"background-color: {bg_color}; border-radius: 5px; padding: 10px;"
    st.markdown(f"<div style='{header_bg}'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2,1])
        
    with col2:
        st.write("")
        
        st.markdown(f'<p style="color:{text_color};font-size:72px;border-radius:2%;text-align:center;">EXIST-Chatbot</p>', unsafe_allow_html=True)


        
    st.markdown("</div>", unsafe_allow_html=True)
    


    colL, colM, colR = st.columns([1,1,1])
    with colL:
        st.image(logo1, width=400)
    with colM:
            # Create input field section
        input_bg = f"background-color: {bg_color}; border-radius: 5px; padding: 10px; margin-top: 20px;"
        st.markdown(f"<div style='{input_bg}'>", unsafe_allow_html=True)
  
        st.markdown(f'<p style="color:{text_color};font-size:24px;border-radius:2%;">"Schön, dass ihr an den Themen Exist und Gründung interessiert seid. Ich habe alle Antworten zum Thema EXIST-Stipendium und Antragsstellung. Naja, fast alle. Meine menschlichen Sklaven, äh, Helfer kann ich dann doch noch nicht entbehren. Sagen die zumindest. Naja, findet es selber heraus, und gebt eure Fragen in das Eingabefeld ein."</p>', unsafe_allow_html=True)
        message = st.text_input("")

        st.markdown("</div>", unsafe_allow_html=True)
        
    

        empty_slot = st.markdown('')
    with colR:
        col4, col5 = st.columns([1,1])
        with col5:
            note = st.image(notice, width=300)
        
   
        


    # Create footer section
    footer_bg = f"background-color: #FFFFF; border-radius: 5px; padding: 10px; margin-top: 20px;"
    st.markdown(f"<div style='{footer_bg}'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([1, 1,1,1])
    with col1:
        st.image(logoHSEL, width=300)
    with col2:
        st.image(logoBMWK, width=300)
    with col3:
        st.image(logoEXIST, width=300)
    with col4:
        st.image(logo2, width=300)
    st.markdown("</div>", unsafe_allow_html=True)
    
    output = ''
        # Check if the button is pressed
    if message:
        keyInt = keyInt + 1
        ai_question = message
        #Greife den Eintrag ab, der am meisten Änhlichkeit mit der Frage hat
        res = search_docs(df_try, ai_question, top_n=1)
        #Greife den Inhalt des Eintrages ab
        context= res.CONTENT.values
        
        #Kombiniere den Prompt mit Baisisprompt, dem Inhalt und der Frage
        combined_prompt = initial_prompt + str(context) + output + "Q: " + ai_question
            
        #API-Abfrage
        messages.append(
            {"role": "user", "content": combined_prompt},
        )        
        chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
        )
            
        #Greife Inhalt des Resultat der API Abfrage ab
        ai_response = chat.choices[0].message.content
        output = ai_response
        update_markdown(empty_slot,output)




