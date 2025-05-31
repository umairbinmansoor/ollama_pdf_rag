import streamlit as st
#############################
# Headers and Prompts
#############################
headers = {
    "authorization":st.secrets["GROQ_API_KEY"],
    "content-type":"application/json"
}