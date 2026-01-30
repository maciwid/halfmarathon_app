import os
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import instructor
from typing import Optional, List
from pydantic import BaseModel
from datetime import date
from pycaret.regression import load_model, predict_model  # type: ignore
from langfuse.decorators import observe
from langfuse.openai import OpenAI


load_dotenv()

MODEL_NAME = 'marathon_model'
MARATHON_LENGTH = 21

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])


def validate_json(json_data: dict, required_keys: list) -> list:
    """
    Sprawdza, które wymagane klucze są brakujące lub mają wartość None/pustą.
    
    Args:
        json_data (dict): JSON / dict do sprawdzenia.
        required_keys (list): lista wymaganych kluczy (str).
        
    Returns:
        list: lista brakujących kluczy. Pusta lista = wszystkie ok.
    """
    missing = []
    for key in required_keys:
        if key not in json_data or json_data[key] is None or (isinstance(json_data[key], (str, list, dict)) and not json_data[key]):
            missing.append(key)
    return missing

def list_missing_items(missing_keys: list) -> str:
    # mapowanie kluczy na polskie nazwy
    key_map = {
        "sex": "Płeć",
        "age": "Wiek",
        "time_per_distance": "Czas na dystans"
    }
    # zamiana na polskie nazwy (ignoruje nieznane klucze)
    polish_keys = [key_map[k] for k in missing_keys if k in key_map]
    
    return "Brakuje danych: " + ", ".join(polish_keys)

class TimePerDistance(BaseModel):
    time_in_seconds: Optional[int] = None
    distance_in_km: Optional[int] = None

class Features(BaseModel):
    sex: Optional[str] = None
    age: Optional[int] = None
    runs_professionally: Optional[bool] = None
    time_per_distance: Optional[List[TimePerDistance]] = None

@observe()
def retrieve_structure(text: str) -> dict:
    instructor_openai_client = instructor.from_openai(get_openai_client())
    res = instructor_openai_client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_model=Features,
        messages=[
            {
                "role": "system",
                "content": (
                    "Wyciągasz dane z tekstu. Zwróć JSON w formacie: "
                    "{sex, age, runs_professionally, time_per_distance: [{time_in_seconds, distance_in_km}]}. "
                    "Jeśli jakaś wartość jest nieobecna lub niejasna, użyj null. "
                    "Czas przelicz na sekundy."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
    )

    return res.model_dump()

def safe_retrieve_structure(text: str) -> dict:
    try:
        res = retrieve_structure(text)
    except Exception as e:
        print("Błąd parsowania:", e)
        res = {
            "sex": None,
            "age": None,
            "runs_professionally": None,
            "time_per_distance": []
        }
    return res


@st.cache_data 
def predict_time(inferred_df):
    model = load_model(MODEL_NAME)
    predicted_time = predict_model(model, data = inferred_df)
    return predicted_time

def seconds_to_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

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
        exhaustion_coeff = 1.04
    else:
        exhaustion_coeff = 1.06
    estimated_time = known_time * ((estimated_distance/known_distance)**exhaustion_coeff)
    tempo = (estimated_time/estimated_distance)/60
    return tempo

def parse_data(response):
    '''
    This function parses json returned from LLM to df compatible with trained model
    '''
    tempos = []
    if response["sex"] in ["male", "man", "mężczyzna"]:
        sex = "M"
    elif response["sex"] in ["female", "kobieta"]:
        sex = "K"
    for item in response["time_per_distance"]:
        tempos.append(estimate_tempo(MARATHON_LENGTH, item["time_in_seconds"], item["distance_in_km"], response["runs_professionally"]))
    if tempos:
        tempo = sum(tempos) / len(tempos) # calculate mean tempo for all given times
    else:
        tempo = np.nan
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
            'Tempo': tempo
        }
    ])
    return inferred_df


# Session state variable init

if "description" not in st.session_state:
    st.session_state["description"] = None

if "response_json" not in st.session_state:
    st.session_state["response_json"] = None


# OpenAI API key protection
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in os.environ:
        st.session_state["openai_api_key"] = os.environ["OPENAI_API_KEY"]

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

with st.form("user_form"):
    st.write("Dzięki modelowi wytrenowanemu na danych z półmaratonu Wrocławskiego jesteśmy w stanie obliczyć szacowany czas na bazie Twoich informacji, takich jak wiek, płeć, czy wyniki w biegu. ")
    st.write("Nie przejmuj się jeśli nie wiesz ile biegniesz 20 km. Program policzy tempo dla dłuższych dystansów na bazie krótszych, aplikując formułę Riegla uwzględniającą zmęczenie. Możesz podać kilka rekordów dla dokładniejszego wyniku")
    user_text = st.text_area("Opowiedz nam o sobie. Ile masz lat? Jakiej jesteś płci? Jakie czasy osiągasz w biegu na długie dystanse (1km+)?")
    submitted = st.form_submit_button("Sprawdź")

if submitted:
    st.session_state["description"] = user_text
    with st.spinner("Poczekaj, aż nasz model przeliczy Twój czas...", show_time=True):
        st.session_state["response_json"] = safe_retrieve_structure(st.session_state["description"])
        missing_keys = validate_json(st.session_state["response_json"], ["sex", "age", "time_per_distance"])
        if missing_keys:
            st.session_state["response_json"] = None
            st.error(f"Dane nie są wystarczające do analizy.") 
            st.info(list_missing_items(missing_keys))
        else:
            st.success("Dane zostały wprowadzone poprawnie")
            if st.session_state["response_json"]["age"] > 99:
                st.info(f"Masz {st.session_state['response_json']['age']} lat? Cóż... Gratulujemy zdrowia!")
            if st.session_state["response_json"]:
                inferred_df = parse_data(st.session_state["response_json"])
                predicted_time = predict_time(inferred_df)
                st.metric("Świetnie! Z naszych obliczeń wynika, że Twój czas na półmaratonie może wynieść około:", seconds_to_time(predicted_time["prediction_label"]))

