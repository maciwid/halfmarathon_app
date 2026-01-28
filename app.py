import streamlit as st
import enum
from dotenv import dotenv_values
from openai import OpenAI
import instructor
from pydantic import BaseModel


if "description" not in st.session_state:
    st.session_state["description"] = None

if "is_description_valid" not in st.session_state:
     st.session_state["is_description_valid"] not in st.session_state

if "response_json" not in st.session_state:
    st.session_state["response_json"] = None


### MAIN ###

st.title("A jaki czas Ty możesz mieć na półmaratonie?")

st.session_state["description"] = st.text_area(
            "Powiedz nam o sobie. Jak szybko potrafisz przebiec 1km? 5km?", 
            height=300, 
        )

if st.session_state["description"]: 
        if st.button("Wyślij zapytanie"):
            st.info("Zapytanie wysłane")  
            try:
                transcript = create_transcription(open(st.session_state["audio_file_path"], "rb").read()) 
            except AuthenticationError:
                info_transcribe_placeholder.error("Invalid API key. Please check your OpenAI API key and try again (refresh site).")
                st.stop()
            except Exception as e:
                info_transcribe_placeholder.error(f"An error occurred: {str(e)}")
                st.stop()
            st.session_state["transcript"] = transcript
            st.rerun()