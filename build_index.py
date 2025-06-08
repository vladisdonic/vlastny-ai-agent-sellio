# build_index.py
import os
import json
import openai
import chromadb
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

# --- API Kľúč vložený priamo do kódu ---
# Toto je v poriadku len pre lokálne spustenie tohto skriptu.
USER_API_KEY = "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"
openai.api_key = USER_API_KEY

if not openai.api_key:
    print("Chyba: OpenAI API kľúč nie je nastavený.")
    exit()

# Nastavenie LlamaIndex Settings
try:
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai.api_key)
    print("Embedding model bol úspešne nastavený.")
except Exception as e:
    print(f"Chyba pri nastavovaní embedding modelu: {e}")
    exit()

# (Funkcie format_parameters, format_request_body, format_responses a load_and_format_openapi_docs zostávajú rovnaké ako predtým)

def format_parameters(parameters):
    """Formátuje sekciu parametrov pre lepšiu čitateľnosť."""
    if not parameters: return "  Žiadne.\n"
    text = ""
    for param in parameters:
        text += f"  - Meno: {param.get('name')}\n"
        text += f"    Umiestnenie: {param.get('in')}\n" 
        param_schema = param.get('schema', {})
        text += f"    Dátový typ: {param_schema.get('type', 'N/A')}\n"
        if 'format' in param_schema: text += f"    Formát: {param_schema.get('format')}\n"
        if 'default' in param_schema: text += f"    Predvolená hodnota: {param_schema.get('default')}\n"
        if 'enum' in param_schema: text += f"    Možné hodnoty (enum): {', '.join(map(str, param_schema.get('enum')))}\n"
        text += f"    Popis: {param.get('description', 'N/A')}\n"
        text += f"    Povinný: {'Áno' if param.get('required') else 'Nie'}\n"
    return text

def format_request_body(request_body):
    """Formátuje sekciu requestBody."""
    if not request_body: return "  Nie je definované.\n"
    text = f"  Popis: {request_body.get('description', 'N/A')}\n"
    text += f"  Povinné: {'Áno' if request_body.get('required') else 'Nie'}\n"
    if request_body.get('content'):
        text += "  Podporované Content-Types a ich schémy:\n"
        for content_type, details in request_body.get('content', {}).items():
            text += f"    - {content_type}:\n"
            if 'schema' in details and '$ref' in details['schema']:
                schema_name = details['schema']['$ref'].split('/')[-1]
                text += f"      Schéma: `{schema_name}` (Odkazuje na definíciu v 'components/schemas')\n"
            elif 'schema' in details:
                 try:
                    text += f"      Schéma: ```json\n{json.dumps(details.get('schema'), indent=4, ensure_ascii=False)}\n```\n"
                 except Exception:
                    text += "      Schéma: (nepodarilo sa naformátovať JSON)\n"
    return text

def format_responses(responses):
    """Formátuje sekciu responses."""
    if not responses: return "  Nie sú definované.\n"
    text = ""
    for status_code, details in responses.items():
        text += f"  - Status Kód: {status_code}\n"
        text += f"    Popis: {details.get('description', 'N/A')}\n"
        if details.get('content'):
            text += "    Podporované Content-Types a ich schémy:\n"
            for content_type, content_details in details.get('content', {}).items():
                text += f"      - {content_type}:\n"
                if 'schema' in content_details and '$ref' in content_details['schema']:
                    schema_name = content_details['schema']['$ref'].split('/')[-1]
                    text += f"        Schéma: `{schema_name}` (Odkazuje na definíciu v 'components/schemas')\n"
                elif 'schema' in content_details:
                    try:
                        text += f"        Schéma: ```json\n{json.dumps(content_details.get('schema'), indent=6, ensure_ascii=False)}\n```\n"
                    except Exception:
                        text += "        Schéma: (nepodarilo sa naformátovať JSON)\n"
    return text

def load_and_format_openapi_docs(file_path="api_docs.json"):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f: openapi_data = json.load(f)
    info = openapi_data.get('info', {}); info_title = info.get('title', 'API Dokumentácia'); info_version = info.get('version', '')
    general_info_content = f"Názov API: {info_title}\nVerzia API: {info_version}\n"
    if 'description' in info: general_info_content += f"Popis API: {info['description']}\n"
    if 'servers' in openapi_data:
        general_info_content += "Servery:\n"
        for server in openapi_data['servers']: general_info_content += f"  - URL: {server.get('url', 'N/A')}, Popis: {server.get('description', 'N/A')}\n"
    documents.append(Document(text=general_info_content, metadata={"type": "general_info", "title": info_title}))
    for path, path_methods in openapi_data.get('paths', {}).items():
        for method_type, details in path_methods.items():
            content = f"API Endpoint: {method_type.upper()} {path}\nSúhrn: {details.get('summary', 'N/A')}\n"
            if details.get('description'): content += f"Popis: {details.get('description')}\n"
            if details.get('tags'): content += f"Tagy: {', '.join(details.get('tags'))}\n"
            if details.get('operationId'): content += f"Operation ID: {details.get('operationId')}\n"
            content += "\nParametre:\n" + format_parameters(details.get('parameters', []))
            content += "\nRequest Body:\n" + format_request_body(details.get('requestBody'))
            content += "\nOdpovede (Responses):\n" + format_responses(details.get('responses', {}))
            content += "\n---\n"
            metadata = {"path": path, "method": method_type.upper(), "summary": details.get('summary', ''), "tags": ", ".join(details.get('tags', []))}
            documents.append(Document(text=content, metadata=metadata))
    if 'components' in openapi_data and 'schemas' in openapi_data['components']:
        for schema_name, schema_details in openapi_data['components']['schemas'].items():
            schema_content = f"Definícia schémy (modelu): {schema_name}\nTyp: {schema_details.get('type', 'N/A')}\n"
            if 'description' in schema_details: schema_content += f"Popis: {schema_details.get('description')}\n"
            if 'properties' in schema_details:
                schema_content += "Vlastnosti (Properties):\n"
                for prop_name, prop_details in schema_details.get('properties', {}).items():
                    prop_type = prop_details.get('type', 'N/A')
                    if '$ref' in prop_details: prop_type = f"referencia na `{prop_details['$ref'].split('/')[-1]}`"
                    elif 'items' in prop_details and '$ref' in prop_details['items']: prop_type = f"pole referencií na `{prop_details['items']['$ref'].split('/')[-1]}`"
                    elif 'items' in prop_details and 'type' in prop_details['items']: prop_type = f"pole typu `{prop_details['items']['type']}`"
                    schema_content += f"  - {prop_name} (typ: {prop_type})"
                    if 'description' in prop_details: schema_content += f": {prop_details['description']}"
                    if 'format' in prop_details: schema_content += f" (formát: {prop_details['format']})"
                    if 'enum' in prop_details: schema_content += f" (možné hodnoty: {', '.join(map(str,prop_details['enum']))})"
                    if 'example' in prop_details: schema_content += f" (príklad: {prop_details['example']})"
                    schema_content += "\n"
            if 'required' in schema_details: schema_content += f"Povinné vlastnosti: {', '.join(schema_details['required'])}\n"
            schema_content += "\n---\n"
            documents.append(Document(text=schema_content, metadata={"type": "schema_definition", "schema_name": schema_name}))
    print(f"Spracovaných {len(documents)} dokumentov z OpenAPI špecifikácie.")
    return documents

if __name__ == "__main__":
    documents = load_and_format_openapi_docs('api_docs.json')
    if not documents: exit()
    db_path = "./chroma_db_sellio"
    collection_name = "api_docs_sellio_v2_9"
    if os.path.exists(db_path):
        import shutil
        shutil.rmtree(db_path)
        print(f"Existujúci priečinok databázy '{db_path}' bol odstránený.")
    os.makedirs(db_path)
    print(f"Vytvorený priečinok pre databázu: {db_path}")
    db = chromadb.PersistentClient(path=db_path)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print(f"Vytváram a ukladám index pre {len(documents)} dokumentov...")
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print("\n--- ÚSPECH! ---")
    print(f"Index bol úspešne vytvorený v priečinku: {db_path}")
    print(f"Celkový počet dokumentov v indexe: {chroma_collection.count()}")
    print("\nTeraz môžete nahrať priečinok 'chroma_db_sellio' na GitHub a spustiť app.py lokálne alebo na cloude.")