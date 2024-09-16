from django.shortcuts import render
from django.http import JsonResponse
# from django.http import HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

import openai
import os
import json
import shutil

import pusher

from .pusher import pusher_client

# Pusher settings
PUSHER_APP_ID = '1599982'
PUSHER_APP_KEY = '3a34edc0fe9c326a3e99'
PUSHER_APP_SECRET = '8996bff39703cd5d2f6c'
PUSHER_APP_CLUSTER = 'sa1'

pusher_client = pusher.Pusher(
    app_id=PUSHER_APP_ID,
    key=PUSHER_APP_KEY,
    secret=PUSHER_APP_SECRET,
    cluster=PUSHER_APP_CLUSTER,
    ssl=True
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from tqdm.auto import tqdm  # progress bar
from io import StringIO

# PDF processing
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as fallback_text_extraction 

# Langchain imports
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma, Pinecone, Qdrant
from langchain.vectorstores import Pinecone
# from langchain_community.embeddings.openai import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings

# import pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.chat_models import ChatOpenAI
from qdrant_client import QdrantClient
from pinecone.core.openapi.shared.exceptions import PineconeApiException

from langchain_pinecone import Pinecone  # Cambiamos la importación
from langchain_pinecone import Pinecone as PineconeVectorStore
# from langchain.vectorstores.pinecone import PineconeVectorStore
# from langchain.vectorstores import Pinecone as PineconeVectorStore
# from langchain_pinecone import PineconeVectorStore

import pinecone  # Asegúrate de importar correctamente el paquete Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings  # Import correcto desde langchain
# from langchain.vectorstores.pinecone import Pinecone as LangchainPinecone
from langchain.chains import load_chain
from langchain.llms import OpenAI


# Inicialización de OpenAI
OPENAI_API_KEY = 'sk-tqTwfvZ9fkkTpi9xC4e2YMFJzNtvRoeW94Ld50WfltT3BlbkFJCX_Fi09aJPqs-t2JAd1xjQgzK81xhkyTe33v8D-W0A'

# Inicialización de Pinecone
PINECONE_API_KEY = 'e1870b8a-1371-4d9f-b4ac-757a9d69b06e'
PINECONE_API_ENV = 'us-east-1'

# Crear una instancia del cliente de Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)

# Embeddings con OpenAI, usando el import correcto desde langchain.embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Definir el nombre del índice
index_name = "chat-documents-ai"

# Verificar si el índice ya existe
existing_indexes = pc.list_indexes()
if index_name not in existing_indexes:
    try:
        pc.create_index(
            name=index_name,
            dimension=1536,  # Dimensiones según tu configuración
            metric='cosine',  # Métrica según tu configuración
            spec=ServerlessSpec(
                cloud='aws',
                region=PINECONE_API_ENV
            )
        )
    except pinecone.exceptions.PineconeApiException as e:
        if e.status == 409:
            print(f"El índice '{index_name}' ya existe.")
        else:
            raise e
else:
    print(f"El índice '{index_name}' ya existe. No se necesita crear uno nuevo.")

# Conectar al índice asegurándonos de que el host sea un string
index = pc.Index(index_name)

# # Inicialización de OpenAI
# OPENAI_API_KEY = 'sk-tqTwfvZ9fkkTpi9xC4e2YMFJzNtvRoeW94Ld50WfltT3BlbkFJCX_Fi09aJPqs-t2JAd1xjQgzK81xhkyTe33v8D-W0A'

# # Inicialización de Pinecone
# PINECONE_API_KEY = 'e1870b8a-1371-4d9f-b4ac-757a9d69b06e'
# PINECONE_API_ENV = 'us-east-1'

# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# # Inicialización de Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Definir el nombre del índice
# index_name = "chat-documents-ai"

# # Verificar si el índice ya existe
# existing_indexes = pc.list_indexes()
# if index_name not in existing_indexes:
#     try:
#         pc.create_index(
#             name=index_name,
#             dimension=1536,  # Dimensiones según tu configuración
#             metric='cosine',  # Métrica según tu configuración
#             spec=ServerlessSpec(
#                 cloud='aws',  # Cloud según tu configuración
#                 region='us-east-1'  # Región según tu configuración
#             )
#         )
#     except PineconeApiException as e:
#         if e.status == 409:
#             print(f"El índice '{index_name}' ya existe.")
#         else:
#             raise e
# else:
#     print(f"El índice '{index_name}' ya existe. No se necesita crear uno nuevo.")

# # Conectar al índice
# index = pc.Index(index_name)

# Definición de funciones:
def obtener_listado_archivos(self):
    directorio = "assets/pdf_files"
    archivos = os.listdir(directorio)
    print(archivos)
    return JsonResponse(archivos, safe=False)

def obtener_listado_archivos_procesados(self):
    directorio = "assets/pdf_files_procesados"
    archivos = os.listdir(directorio)
    print(archivos)
    return JsonResponse(archivos, safe=False)

def conversation(request):
    response_data = {
        'message': 'Hola! Soy Gastón.',
    }
    return JsonResponse(response_data)

# @csrf_exempt
def guardar_archivo_audio(request):
    if request.method == 'POST':
        request.upload_handlers.pop(0)
        audio_file = request.FILES['audio_file']
        nombre_audio = audio_file.name
        file_path = os.path.join('assets', 'audio_files', nombre_audio)

        with open(file_path, 'wb+') as archivo:
            for chunk in audio_file.chunks():
                archivo.write(chunk)

        print("El archivo se ha guardado en:", file_path)

        try:
            # Transcribir el archivo de audio usando la nueva API
            with open(file_path, "rb") as audio_file:
                transcript = openai.Audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            print('transcript:', transcript['text'])

            # Generar una respuesta usando el texto transcrito
            response = openai.Completion.create(
                engine="text-davinci-003",  # Cambia el modelo según tu necesidad
                prompt=transcript['text'],
                max_tokens=1000,
                n=1,
                stop=None,
                temperature=0.5,
            )

            generated_text = response.choices[0].text.strip()
            print('generated_text:', generated_text)

            return JsonResponse({'status': 'Archivo de audio traducido a texto exitosamente.', 'generated_text': generated_text})

        except Exception as e:
            return JsonResponse({'status': 'Error en la transcripción del archivo.', 'error': str(e)}, status=500)

    else:
        return JsonResponse({'status': 'Método de solicitud HTTP no permitido.'}, status=405)
    
    
def subir_pdf(request):
    if request.method == 'POST':
        request.upload_handlers.pop(0)  
        if request.FILES['pdf']:
            pdf_file = request.FILES['pdf']
            print('Nombre del PDF: ' + pdf_file.name)

            with open('assets/pdf_files/' + pdf_file.name, 'wb') as f:
                for chunk in pdf_file.chunks():
                    f.write(chunk)

            return obtener_listado_archivos(self=any)

def entrenar_ia(self):
    directorio = "assets/pdf_files"
    directorio_destino = "assets/pdf_files_procesados"
    texts = []
    
    for archivo in os.listdir(directorio):
        path_completo = os.path.join(directorio, archivo)
        print(path_completo)
        with open(path_completo, "rb") as f:
            try:
                loader = UnstructuredPDFLoader(path_completo)
                data = loader.load()
                print(f'You have {len(data)} document(s) in your data')
                print(f'There are {len(data[0].page_content)} characters in your document')

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                documents = text_splitter.split_documents(data)
                print(f'Now you have {len(documents)} documents')

                # Convertir los textos en vectores
                texts = [doc.page_content for doc in documents]
                embeddings_result = embeddings.embed_documents(texts)
                vectors = [(f"vec{i}", embedding, {"source": archivo}) for i, embedding in enumerate(embeddings_result)]

                # Upsert los vectores al índice de Pinecone
                upsert_response = index.upsert(
                    vectors=vectors,
                    namespace="example-namespace"
                )

                print(f'Upsert response: {upsert_response}')

                source_path = os.path.join(directorio, archivo)
                destination_path = os.path.join(directorio_destino, archivo)
                move_file(source_path, destination_path)

            except Exception as exc: 
                print(f'Exc: {exc}')
                return JsonResponse({'success': False})

    return JsonResponse({'success': True})


def move_file(source_path, destination_path):
    try:
        shutil.move(source_path, destination_path)
        print("Archivo movido exitosamente.")
    except FileNotFoundError:
        print("No se encontró el archivo en la ruta de origen.")
    except PermissionError:
        print("No se tienen los permisos necesarios para mover el archivo.")
    except Exception as e:
        print(f"Ocurrió un error al mover el archivo: {str(e)}")




def consultar_pdf(consulta):
    print(f'Consultar_pdf: {consulta}')
    
    # Listar índices correctamente usando el cliente `pc`
    print('Listando índices...')
    print(pc.list_indexes())  # Ahora listamos los índices a través del cliente Pinecone.
    print(index_name)

    # Describir el índice
    print(index.describe_index_stats())

    text_field = "text"

    # Usamos `PineconeVectorStore` desde `langchain_pinecone`
    vectorstore = PineconeVectorStore(index=index, text_key=text_field, embedding=embeddings)

    # Realizar la búsqueda de similitud por texto en lugar de vector
    vector_result = vectorstore.similarity_search(consulta, k=3)

    print('vector_result')
    print(vector_result)

    # Usar el modelo de OpenAI
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    
    # Cargar la cadena de preguntas y respuestas
    chain = load_qa_chain(llm, chain_type="stuff")
    text_response_ai = chain.run(input_documents=vector_result, question=consulta)
    
    print(text_response_ai)

    # Enviar la respuesta usando pusher
    pusher_client.trigger('chat', 'message', {
        'theme': 'is-link',
        'username': 'Conversation UI',
        'message': text_response_ai
    })

    return JsonResponse({'success': True, 'text_response_ai': text_response_ai})

@csrf_exempt
def send_pusher_event(request):
    request.upload_handlers.pop(0)    
    data = json.loads(request.body)
    if request.method == 'POST':
        consultar_pdf(data['consulta'])

    return JsonResponse({'success': True})

