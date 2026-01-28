import streamlit as st
import enum
import pandas as pd
import numpy as np
from dotenv import dotenv_values, load_dotenv
from openai import OpenAI
import instructor
from pydantic import BaseModel
import boto3
import os
from datetime import date
from pycaret.regression import load_model, predict_model  # type: ignore

env = dotenv_values(".env")

MODEL_NAME = 'marathon_model'
MARATHON_LENGTH = 21

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

class ClassificationCategory(str, enum.Enum):
    VALID = "valid"
    NOT_VALID= "not_valid"

def check_validity(description):
    class ClassificationResult(BaseModel):
        category: ClassificationCategory
    instructor_openai_client = instructor.from_openai(get_openai_client())
    res = instructor_openai_client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=ClassificationResult,
        messages=[
            {
                "role": "user",
                "content": f"Sprawdź czy tekst zawiera KAŻDĄ z tych informacji: płeć && wiek && czas w bieganiu: {description}",
            },
        ],
    )
    return res.category

def retrieve_structure(text):
    class TimePerDistance(BaseModel):
        time_in_seconds: int      # np. "12:30"
        distance_in_km: int 
    class Features(BaseModel):
        sex: str
        age: int
        runs_professionally: bool
        time_per_distance: list[TimePerDistance]
    instructor_openai_client = instructor.from_openai(get_openai_client())
    res = instructor_openai_client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_model=Features,
        messages=[
            {
                "role": "user",
                "content": text,
            },
        ],
    )

    return res.model_dump()

def estimate_tempo(
        estimated_distance, 
        known_time,
        known_distance,
        runs_professionally = False
        ):
    """
    This function estimates time for a given distance using Riegel formula
    """
    if runs_professionally:
        exhaustion_coeff = 1.04,
    else:
        exhaustion_coeff = 1.06
    estimated_time = known_time * ((estimated_distance/known_distance)**exhaustion_coeff)
    tempo = (estimated_time/estimated_distance)/60
    return tempo

@st.cache_data 
def load_model():
    return load_model(MODEL_NAME)

def parse_data(response):
    tempos = []
    if response["sex"] == "male":
        sex = "M"
    elif response["sex"] == "female":
        sex = "K"
    for item in response["time_per_distance"]:
        tempos.append(estimate_tempo(MARATHON_LENGTH, item["time_in_seconds"], item["distance_in_km"], response["runs_professionally"]))
    tempo = sum(tempos) / len(tempos)
    rocznik = date.today().year - response["age"]
    age_category = (response["age"] // 10) * 10
    sex_age_category = f"{sex}{age_category}"
    inferred_df = pd.DataFrame([
        {
            'Miejsce': np.nan,
            'Numer startowy': np.nan,
            'Imię': np.nan,
            'Nazwisko': np.nan,
            'Miasto': np.nan,
            'Kraj': np.nan,
            'Drużyna': np.nan,
            'Płeć': sex,
            'Płeć Miejsce': np.nan,
            'Kategoria wiekowa': sex_age_category,
            'Kategoria wiekowa Miejsce': np.nan,
            'Rocznik': rocznik, 
            '5 km Czas': np.nan,
            '5 km Miejsce Open': np.nan,
            '5 km Tempo': np.nan,
            '10 km Czas': np.nan,
            '10 km Miejsce Open': np.nan,
            '10 km Tempo': np.nan,
            '15 km Czas': np.nan,
            '15 km Miejsce Open': np.nan,
            '15 km Tempo': np.nan,
            '20 km Czas': np.nan,
            '20 km Miejsce Open': np.nan,
            '20 km Tempo': np.nan,
            'Tempo Stabilność': np.nan,
            'Czas': np.nan,
            'Tempo': tempo
        }
    ])
    return inferred_df


# Session state variable init

if "description" not in st.session_state:
    st.session_state["description"] = None

if "is_description_valid" not in st.session_state:
     st.session_state["is_description_valid"] = None

if "response_json" not in st.session_state:
    st.session_state["response_json"] = None


# OpenAI API key protection
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.info("Dodaj swój klucz Open API, aby korzystać z aplikacji")
        st.page_link("https://platform.openai.com/account/api-keys", label="Zdobądź swój klucz tutaj", help= "Nie masz jeszcze klucza?", icon="🔑")
        st.session_state["openai_api_key"] = st.text_input("klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

### MAIN ###

st.title("A jaki czas Ty możesz mieć na półmaratonie?")

st.session_state["description"] = st.text_area(
            "Opowiedz nam o sobie. Podziel się osiągnięciami w biegu na długie dystanse (1km+)", 
            height=300, 
        )

if st.session_state["description"]: 
        if st.button("Wyślij zapytanie"):
            st.session_state["is_description_valid"] = None
            try:
                if check_validity(st.session_state["description"]) == "valid":
                    st.session_state["is_description_valid"] = True
                    st.write("Dobre dane")
                else:
                    st.session_state["is_description_valid"] = False
                    st.session_state["response_json"] = None
                    st.write("Dane nie są wystarczające do analizy. Najlepiej aby podany opis zawierał informację o wieku, płci i czasie na dowolny dłuszy dystans") 
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()
            if st.session_state["is_description_valid"]:
                st.session_state["response_json"] = retrieve_structure(st.session_state["description"])
                st.write(st.session_state["response_json"])

if st.session_state["response_json"]:
    inferred_df = parse_data(st.session_state["response_json"])
    # model = load_model()
    # predicted_time = predict_model(model, inferred_df)
    # st.write(predicted_time)
    st.dataframe(inferred_df)