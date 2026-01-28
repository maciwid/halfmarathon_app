import streamlit as st
import enum
from dotenv import dotenv_values
from openai import OpenAI
import instructor
from pydantic import BaseModel

env = dotenv_values(".env")

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
    class Features(BaseModel):
        sex: str
        age: int
        time_per_distance: list[str, str]
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
            try:
                if check_validity(st.session_state["description"]) == "valid":
                    st.session_state["is_description_valid"] = True
                    st.write("Dobre dane")
                else:
                    st.write("Dane nie są wystarczające do analizy. Najlepiej aby podany opis zawierał informację o wieku, płci i czasie na dowolny dłuszy dystans") 
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()
            if st.session_state["is_description_valid"]:
                st.write(retrieve_structure(st.session_state["description"]))
            