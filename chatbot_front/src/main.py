import os
import requests
import streamlit as st
import time

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/chatbot-rag-agent")
assistant_icon = "assets/zari_ico.png"

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Configurar el tema de la aplicación
st.set_page_config(
    page_title="Demo de Zaribot",
    page_icon="assets/zari.png",
    layout="centered",
    initial_sidebar_state="expanded",
)

local_css("assets/style.css")

with st.sidebar:
    st.header("Acerca del asistente")
    st.markdown(
        """
        Esta es la interfaz de un chatbot con
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agente diseñado para responder preguntas sobre el portal web de la Universidad del Cauca.
        El agente utiliza generación aumentada por recuperación (RAG) sobre datos
        tanto estructurados como no estructurados que han sido proveídos por las TIC.
        (Esto es solo una demo)
        """
    )

    st.header("ACLARACIÓN")
    st.markdown(
        """
        Lo que se muestra aquí es una demostracion y puede no representar el producto final, el
        objetivo de esta demo es mostrar algunas de las capacidades del asistente.
        """
    )

# Añade un contenedor para el título y la imagen
with st.container():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("assets/zari.png", width=75)
    with col2:
        st.title("Demo de Zaribot")

st.info(
    "Hazme preguntas sobre dependencias y divisiones"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "saludo_mostrado" not in st.session_state:
    requests.post(CHATBOT_URL, json={"text": "Tu nombre sera Zaribot y todas tus respuestas seran en español"})
    time.sleep(10)

    response = requests.post(CHATBOT_URL, json={"text": "Hola, saludame con tu nombre y dime cual es tu funcion"})
    output_text = response.json()["output"]

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
        }
    )

    st.session_state.saludo_mostrado = True

for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["output"]}</div>', unsafe_allow_html=True)
        elif message["role"] == "assistant":
            with st.container():
                col1, col2 = st.columns([1, 20])
                with col1:
                    st.image("assets/zari_ico.png", width=35)
                with col2:
                    st.markdown(f'<div class="assistant-message"> {message["output"]}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("¿Qué deseas saber?"):
    st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner("Buscando una respuesta..."):
        response = requests.post(CHATBOT_URL, json=data)

        if response.status_code == 200:
            output_text = response.json()["output"]
        else:
            output_text = "Se produjo un error al procesar tu mensaje. Por favor, inténtalo de nuevo o reformula tu mensaje."

    with st.container():
        col1, col2 = st.columns([1, 20])
        with col1:
            st.image("assets/zari_ico.png", width=35)
        with col2:
            st.markdown(f'<div class="assistant-message">{output_text}</div>', unsafe_allow_html=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
        }
    )
