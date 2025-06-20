import streamlit as st 
import utils 


st.set_page_config(page_title= "ChatBot ", page_icon="", layout="wide")

st.title("ChatBot Basico 166")

#historial

if "history" not in st.session_state:
    st.session_state.history = []

#contexto 
if "context" not in st.session_state:
    st.session_state.context = []

#Construimos el espacio, emisor-mensaje
for sender, msg in st.session_state.history:
    if sender == "Tu":
        st.markdown(f'**🐱‍🐉{sender}:**{msg}')
    else:
        st.markdown(f'**🐱‍👤{sender}:** {msg}')

#Sino hay entrada
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

#Procesamiento de la entrada
def send_msg():
    user_input = st.session_state.user_input.strip()
    if user_input:
        tag = utils.predict_class(user_input)
        st.session_state.context.append(tag)
        response = utils.get_response(tag, st.session_state.context)
        st.session_state.history.append(('Tu', user_input))
        st.session_state.history.append(('Bot', response))
        st.session_state.user_input = ""

#Creamos el campo de texto
st.text_input("Escribe tu mensaje:", key="user_input", on_change=send_msg)





