
import streamlit as st
import openai
import os
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
import json

st.set_page_config(page_title="API Asistent Sellio", layout="wide", page_icon="🤖")

def check_password():
    correct_password = st.secrets.get("APP_PASSWORD", "predvoleneHeslo123")
    if st.session_state.get("password_correct", False):
        return True
    
    st.header("🔑 Prístup k aplikácii")
    password_placeholder = st.empty()
    password = password_placeholder.text_input("Zadajte prístupové heslo:", type="password", key="password_input")

    if password:
        if password == correct_password:
            st.session_state.password_correct = True
            password_placeholder.empty()
            st.rerun()
            return True
        else:
            st.error("Nesprávne heslo. Skúste znova.")
            st.session_state.password_correct = False
            return False
    return False

if not check_password():
    st.stop()

st.sidebar.success("Prístup povolený.", icon="✅")

# Načítanie a nastavenie API kľúča
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_key = api_key
    st.sidebar.info("API kľúč načítaný zo Streamlit Secrets.", icon="☁️")
except (AttributeError, KeyError, FileNotFoundError):
    st.error("Chyba konfigurácie: OpenAI API kľúč nebol nájdený v Streamlit Secrets.")
    st.stop()
    
# Nastavenie modelov pre LlamaIndex
try:
    Settings.llm = OpenAI(model="gpt-4o")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
except Exception as e:
    st.error(f"Chyba pri nastavovaní LlamaIndex modelov: {e}")
    st.stop()

st.title("🤖 Váš Interný API Asistent")
st.markdown("Opýtajte sa ma čokoľvek o našej API. Odpoviem na základe **Sellio 2 API dokumentácie**.")
st.markdown("---")

@st.cache_resource(show_spinner="Pripájam sa k znalostnej databáze...")
def load_index():
    db_path = "./chroma_db_sellio_v2" # Nový názov priečinka
    collection_name = "api_docs_sellio_latest" # Nový názov kolekcie
    
    if not os.path.exists(db_path) or not os.listdir(db_path):
        st.error(f"Chyba: Priečinok databázy '{db_path}' nebol nájdený. Uistite sa, že je nahratý na GitHub.")
        st.stop()
        
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    
    if chroma_collection.count() == 0:
        st.error(f"Chyba: Databáza '{collection_name}' je prázdna.")
        st.stop()

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    st.sidebar.info(f"Úspešne pripojený k databáze '{collection_name}'.", icon="📚")
    return index

try:
    index = load_index()
except Exception as e:
    st.error(f"Nastala chyba pri načítavaní databázy: {e}")
    st.stop()

if "chat_engine" not in st.session_state:
    system_prompt_text = (
        "Si expertný AI asistent pre API dokumentáciu 'Sellio 2'. "
        "Tvojou úlohou je presne a stručne odpovedať na otázky týkajúce sa tejto API na základe poskytnutého kontextu z dokumentácie. "
        "Vždy odpovedaj v slovenčine. "
        "Ak informácia nie je v kontexte, povedz, že ju nemáš k dispozícii, nevymýšľaj si. "
        "Pri popisovaní endpointov uveď HTTP metódu a cestu (napr. 'POST /api/v1/products')."
    )
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        system_prompt=system_prompt_text,
        verbose=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Dobrý deň! Som váš AI asistent pre Sellio 2 API. Ako vám môžem pomôcť?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Vaša otázka k API..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 AI premýšľa..."):
            try:
                response = st.session_state.chat_engine.chat(user_query)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})
            except Exception as e:
                error_msg = f"Prepáčte, nastala chyba: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if st.sidebar.button("Vymazať históriu chatu"):
    st.session_state.messages = [{"role": "assistant", "content": "Dobrý deň! Som váš AI asistent pre Sellio 2 API. Ako vám môžem pomôcť?"}]
    if "chat_engine" in st.session_state:
        st.session_state.chat_engine.reset()
    st.session_state.password_correct = False
    st.rerun()
