# app.py
import streamlit as st
import openai
import os 
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
import json

# --- Konfigurácia stránky ---
st.set_page_config(page_title="API Asistent Sellio", layout="wide", page_icon="🤖")

# --- Funkcia na kontrolu hesla ---
def check_password():
    """Vráti True, ak používateľ zadal správne heslo, inak False."""
    # Načítanie hesla zo Streamlit Secrets
    # V .streamlit/secrets.toml by ste mali mať: APP_PASSWORD = "vase_super_tajne_heslo"
    correct_password = st.secrets.get("APP_PASSWORD", "predvoleneHeslo123") # Fallback pre lokálne testovanie bez secrets

    # Ak už bolo heslo zadané a je správne, uložíme si to do session state
    if st.session_state.get("password_correct", False):
        return True

    # Zobrazenie formulára na zadanie hesla
    st.header("🔑 Prístup k aplikácii")
    password_placeholder = st.empty()
    password = password_placeholder.text_input("Zadajte prístupové heslo:", type="password", key="password_input")

    if password:
        if password == correct_password:
            st.session_state.password_correct = True
            password_placeholder.empty() # Odstráni input pole po správnom zadaní
            st.rerun() # Znovu načíta stránku, teraz už s prístupom
            return True
        else:
            st.error("Nesprávne heslo. Skúste znova.")
            st.session_state.password_correct = False
            return False
    return False

# --- Funkcie na formátovanie (zostávajú rovnaké) ---
def pretty_print_parameters(params):
    if not params:
        return "Žiadne."
    output = ""
    for p in params:
        output += f"- **{p.get('name')}** ({p.get('in')}, {p.get('schema', {}).get('type', 'N/A')})"
        if p.get('required'):
            output += " (povinný)"
        output += f": {p.get('description', 'Bez popisu.')}\n"
    return output

def pretty_print_request_body(body):
    if not body or not body.get('content'):
        return "Nie je definované."
    output = ""
    for content_type, details in body.get('content', {}).items():
        output += f"  - **{content_type}**:\n"
        if 'schema' in details and '$ref' in details['schema']:
            schema_name = details['schema']['$ref'].split('/')[-1]
            output += f"    - Schéma: `{schema_name}` (Detail nájdete v sekcii 'components/schemas' vašej OpenAPI špecifikácie)\n"
        elif 'schema' in details:
            try:
                output += f"    - Schéma: ```json\n{json.dumps(details.get('schema'), indent=2, ensure_ascii=False)}\n```\n"
            except Exception:
                output += "    - Schéma: (nepodarilo sa naformátovať)\n"
    return output

# --- Hlavná logika aplikácie ---
if not check_password():
    st.stop() # Zastaví vykonávanie zvyšku aplikácie, ak heslo nie je správne

# Ak sme sa dostali sem, heslo bolo správne zadané
st.sidebar.success("Prístup povolený.", icon="✅")

# --- Bezpečné načítanie API kľúča zo Streamlit Secrets ---
# Na cloude MUSÍ byť API kľúč v Streamlit Secrets
# Lokálne môže byť stále ako environmentálna premenná pre testovanie
api_key = None
AI_ENABLED = False
try:
    api_key = st.secrets["OPENAI_API_KEY"] # Primárne pre cloud
    if api_key:
        st.sidebar.info("API kľúč načítaný zo Streamlit Secrets.", icon="☁️")
        AI_ENABLED = True
except (AttributeError, KeyError, FileNotFoundError): # AttributeError ak st.secrets neexistuje (lokálne), KeyError ak kľúč chýba
    st.sidebar.warning("Streamlit Secrets pre OpenAI API kľúč nenájdené. Skúšam lokálne prostredie...")
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        st.sidebar.info("API kľúč načítaný z lokálneho prostredia (pre testovanie).", icon="💻")
        AI_ENABLED = True

if not AI_ENABLED or not api_key:
    st.error("Chyba konfigurácie: OpenAI API kľúč nebol nájdený. Uistite sa, že je nastavený v Streamlit Secrets pre túto aplikáciu.")
    st.stop()
    
openai.api_key = api_key

# <<< Nastavenie LlamaIndex Settings >>>
try:
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai.api_key)
    # Ak používate LLM od OpenAI aj pre chat, môžete ho nastaviť tu globálne:
    # from llama_index.llms.openai import OpenAI
    # Settings.llm = OpenAI(model="gpt-4o", api_key=openai.api_key) 
except Exception as e:
    st.error(f"Chyba pri nastavovaní LlamaIndex Settings (embed_model/llm): {e}")
    st.stop()

# --- Hlavný titulok a popis aplikácie ---
st.title("🤖 Váš Interný API Asistent")
st.markdown("Opýtajte sa ma čokoľvek o našej API. Odpoviem na základe **Sellio 2 API dokumentácie** (verzia 2.9).")
st.markdown("---")

# --- Funkcia na načítanie existujúceho indexu z ChromaDB ---
@st.cache_resource(show_spinner="Pripájam sa k znalostnej databáze...")
def load_index_from_chroma():
    db_path = "./chroma_db_sellio" 
    if not os.path.exists(db_path) or not os.listdir(db_path):
        st.error(f"Chyba: Priečinok '{db_path}' pre ChromaDB nebol nájdený alebo je prázdny. Uistite sa, že je súčasťou vášho projektu na GitHube a že ste predtým lokálne spustili `build_index.py`.")
        st.stop()
        
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection_name = "api_docs_sellio_v2_9"
    
    try:
        chroma_collection = db.get_collection(name=chroma_collection_name)
    except Exception as e: 
        st.error(f"Chyba: Kolekcia '{chroma_collection_name}' v ChromaDB nebola nájdená. Uistite sa, že `build_index.py` prebehol správne. Detail: {e}")
        st.stop()

    if chroma_collection.count() == 0:
        st.error(f"Chyba: Kolekcia '{chroma_collection.name}' v ChromaDB je prázdna. Uistite sa, že `build_index.py` prebehol správne a naplnil databázu.")
        st.stop()

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    # LlamaIndex by mal automaticky použiť model z globálnych Settings
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    st.sidebar.info(f"Úspešne pripojený k databáze '{chroma_collection.name}' ({chroma_collection.count()} dokumentov).", icon="📚")
    return index

# Načítanie indexu
try:
    index = load_index_from_chroma()
except Exception as e:
    st.error(f"Nastala chyba pri načítavaní indexu z ChromaDB: {e}")
    st.stop()

# --- Nastavenie Chat Engine ---
if "chat_engine" not in st.session_state:
    system_prompt_text = (
        "Si expertný AI asistent pre API dokumentáciu 'Sellio 2' (verzia 2.9). "
        "Tvojou úlohou je presne a stručne odpovedať na otázky týkajúce sa tejto API na základe poskytnutého kontextu z dokumentácie. "
        "Vždy odpovedaj v slovenčine. "
        "Ak informácia nie je v kontexte, povedz, že ju nemáš k dispozícii, nevymýšľaj si. "
        "Pri popisovaní endpointov uveď HTTP metódu a cestu (napr. 'POST /api/v1/products'). "
        "Ak otázka smeruje k parametrom, request body alebo responses, snaž sa ich pekne naformátovať."
        "Ak je v kontexte kódový príklad (JSON schema), môžeš ho zahrnúť do odpovede v Markdown code blocku."
        "Buď nápomocný a priateľský."
    )
    
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context", 
        system_prompt=system_prompt_text,
        # Ak ste nastavili Settings.llm globálne, nemusíte ho tu špecifikovať
        # llm=OpenAI(model="gpt-4o", api_key=openai.api_key), # Ak chcete explicitne pre chat engine
        verbose=True 
    )

# --- História chatu ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Dobrý deň! Som váš AI asistent pre Sellio 2 API. Ako vám môžem pomôcť?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Vstup od používateľa a generovanie odpovede ---
if user_query := st.chat_input("Vaša otázka k API..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("🤖 AI premýšľa..."):
            try:
                response = st.session_state.chat_engine.chat(user_query)
                ai_response_content = response.response
            except Exception as e:
                ai_response_content = f"Prepáčte, nastala chyba pri spracovaní vašej požiadavky: {e}"
                st.error(ai_response_content)

        message_placeholder.markdown(ai_response_content)
        st.session_state.messages.append({"role": "assistant", "content": ai_response_content})

# --- Sidebar s informáciami ---
st.sidebar.header("O aplikácii")
st.sidebar.markdown("""
Tento AI asistent je navrhnutý tak, aby vám pomohol rýchlo nájsť informácie 
v API dokumentácii pre **Sellio 2 (verzia 2.9)**.
""")
st.sidebar.markdown("Použité technológie: Streamlit, LlamaIndex, OpenAI, ChromaDB.")

if st.sidebar.button("Vymazať históriu chatu"):
    st.session_state.messages = [{"role": "assistant", "content": "Dobrý deň! Som váš AI asistent pre Sellio 2 API. Ako vám môžem pomôcť?"}]
    if "chat_engine" in st.session_state:
        st.session_state.chat_engine.reset()
    st.session_state.password_correct = False # Aby si pýtalo heslo znova
    st.rerun()
