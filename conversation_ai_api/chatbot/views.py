from django.shortcuts import render
from django.http import JsonResponse
# from django.http import HttpResponse
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

import openai
# import whisper
# import io
import os
import json
import shutil

import pusher

from .pusher import pusher_client
PUSHER_APP_ID = '1599982'
PUSHER_APP_KEY = '3a34edc0fe9c326a3e99'
PUSHER_APP_SECRET = '8996bff39703cd5d2f6c'
PUSHER_APP_CLUSTER = 'sa1'

pusher_client = pusher.Pusher(
app_id=PUSHER_APP_ID,
key=PUSHER_APP_KEY,
secret=PUSHER_APP_SECRET,
cluster=PUSHER_APP_CLUSTER,
ssl=True)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"

# from gensim.models import Word2Vec

# from datasets import load_dataset

# from sentence_transformers import SentenceTransformer, util

from tqdm.auto import tqdm  # progress bar


# import pandas as pd
from io import StringIO

# import tabula

# import pdftables_api
# conversion = pdftables_api.Client('wyyx39rh1cny')

# import pytesseract
# import PyPDF2
# from PIL import Image
from PyPDF2 import PdfReader 
from pdfminer.high_level import extract_text as fallback_text_extraction 


# Langchain
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone, Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import CohereEmbeddings
from qdrant_client import QdrantClient
# from qdrant_client import QdrantClient


# Inicialización de openAI
openai.api_key = "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"

# Inicialización de Pinecone
OPENAI_API_KEY = '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'
PINECONE_API_KEY = '339c3992-2c17-4d86-a097-17c1140f1d71'
PINECONE_API_ENV = 'us-west1-gcp-free'


# Inicialización de Cohere
cohere_api_key = 'bLKANujveQGajetRNlNLxTFGZMJ8OpFP61fdL5MG'
qdrant_url = 'https://046ffe43-cabf-4d01-b0a3-36fb0b12ce9a.us-east-1-0.aws.cloud.qdrant.io:6333'
qdrant_api_key = 'KzTbvZ_ZHHyYH9XM3DvaV2DR_vr8Wj8hlEKhQUByTktsiTN40Wz62g'

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)

index_name = "langchain2-index"

# Definición de funciones:

def obtener_listado_archivos(self):
    directorio = "assets/pdf_files"
    archivos = os.listdir(directorio)
    print(archivos)

    # archivos_json = json.dumps(archivos)
    # print(archivos_json)
    # return json.dumps(archivos)

    return JsonResponse(archivos, safe=False)

def obtener_listado_archivos_procesados(self):
    directorio = "assets/pdf_files_procesados"
    archivos = os.listdir(directorio)
    print(archivos)

    # archivos_json = json.dumps(archivos)
    # print(archivos_json)
    # return json.dumps(archivos)

    return JsonResponse(archivos, safe=False)

def obtener_listado_archivos_cohere(self):
    directorio = "assets/pdf_files_cohere"
    archivos = os.listdir(directorio)
    print(archivos)

    # archivos_json = json.dumps(archivos)
    # print(archivos_json)
    # return json.dumps(archivos)

    return JsonResponse(archivos, safe=False)

def obtener_listado_archivos_cohere_procesados(self):
    directorio = "assets/pdf_files_cohere_procesados"
    archivos = os.listdir(directorio)
    print(archivos)

    # archivos_json = json.dumps(archivos)
    # print(archivos_json)
    # return json.dumps(archivos)

    return JsonResponse(archivos, safe=False)


def conversation(request):
    # Aquí debe agregar la lógica de su chatbot
    response_data = {
        'message': 'Hola! Soy Gastón.',
    }
    return JsonResponse(response_data)


def guardar_archivo_audio(request):
    if request.method == 'POST':

        request.upload_handlers.pop(0)

        audio_file = request.FILES['audio_file']
        print('Nombre del audio: ' + audio_file.name)

        nombre_audio = audio_file.name + '.wav'

        print('nombre_audio: ' + nombre_audio)

        # Guardar archivo en el proyecto

        file_path = 'assets/audio_files/' + nombre_audio

        print('file_path: ' + file_path)
        
        with open(file_path, 'wb+') as archivo:
            for chunk in audio_file.chunks():
                archivo.write(chunk)
                

        print("El archivo se ha guardado en:", file_path)
        
        # Obtener texto del audio Con Whisper de manera local: (Más lento)
        # whisper_model = whisper.load_model("medium")
        # result = whisper_model.transcribe(file_path)
        # text_result = result["text"]
        # print('result: ' + result["text"])

        # Obtener texto del audio Con Whisper desde la api de openAi: (Mucho más rápido)
        audio_file= open(file_path, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

        print('transcript')
        print(transcript)
        # Llamar a la API de OpenAI utilizando el texto generado por el audio como prompt
        model = "text-davinci-002"
        response = openai.Completion.create(
            engine=model,
            prompt=transcript["text"],
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.5,
        )

        # Obtener el texto generado por el modelo de OpenAI
        generated_text = response.choices[0].text

        # Formatear: Porque aparecen 4 caracteres antes en su respuesta
        # formatted_generated_text = generated_text[2:len(generated_text)]

        # Responder a la interfaz gráfica la respuesta generada por openAi
        return JsonResponse({'status': 'Archivo de audio traducido a texto exitosamente.', 'generated_text': generated_text})
    else:
        return JsonResponse({'status': 'Método de solicitud HTTP no permitido.'}, status=405)


def subir_pdf(request):
    if request.method == 'POST':

        request.upload_handlers.pop(0)  
        print('subir_pdf')

        if request.method == 'POST' and request.FILES['pdf']:
            pdf_file = request.FILES['pdf']
            print('Nombre del PDF: ' + pdf_file.name)

        with open('assets/pdf_files/' + pdf_file.name, 'wb') as f:
                for chunk in pdf_file.chunks():
                    f.write(chunk)

        return (obtener_listado_archivos(self=any))


def subir_pdf_cohere(request):
    if request.method == 'POST':
        request.upload_handlers.pop(0)  
        print('subir_pdf_cohere')

        if request.method == 'POST' and request.FILES['pdf']:
            pdf_file = request.FILES['pdf']
            print('Nombre del PDF: ' + pdf_file.name)

        with open('assets/pdf_files_cohere/' + pdf_file.name, 'wb') as f:
                for chunk in pdf_file.chunks():
                    f.write(chunk)

        return (obtener_listado_archivos_cohere(self=any))





def entrenar_ia(self):
    # print('hola')
    directorio = "assets/pdf_files"
    directorio_destino = "assets/pdf_files_procesados"
    # text=""
    texts=""
    text=""
    # Recorrer los archivos del directorio
    for archivo in os.listdir(directorio):
        # Obtener el directorio de cada archivo
        path_completo= os.path.join(directorio, archivo)
        print(path_completo)
        print(archivo)
        print('Hola<1')
        with open(path_completo, "rb") as f:
            try: 
                print('Hola')

                # Para utilizar con Pinecone
                loader = UnstructuredPDFLoader(path_completo)
                data = loader.load()
                print('data')
                print(data)
                print (f'You have {len(data)} document(s) in your data')
                print (f'There are {len(data[0].page_content)} characters in your document')

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(data)
                print('texts')
                print(texts)
                
                print (f'Now you have {len(texts)} documents')


                # Utilizar Pinecone para guardar los pdf nuevos y entrenar la IA:
                # print('embeddings')
                # print(embeddings)
                docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)


                # docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings_new, index_name=index_name)
                # docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)
                # print('docsearch')
                # print(docsearch)
                # 'Qué es una organización?'
                # 'Cómo es la embriogénesis?'
                # 'Quién fue el padre de jesús?'
                # query = "cómo es la medicina natural?"

                # docs = docsearch.similarity_search(query, include_metadata=True)
                # print('docs:')
                # print(docs)
                




                # # Conectar a la api de openAi y enviar docs generados por Pinecone 
                # llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
                # chain = load_qa_chain(llm, chain_type="stuff")
                
                # text_response_ai = chain.run(input_documents=docs, question=query)

                # print(text_response_ai)  








                # Convertir pdf a csv con pdf_tables (No funciona bien, no acepta el pdf)
                # archivo_nuevo = archivo + '.csv'
                # path_completo_nuevo = os.path.join(directorio, archivo_nuevo)
                
                # conversion.csv(path_completo, path_completo_nuevo)



                # Convertir pdf a csv con tabula (Emite csv en blanco)
                # Read PDF File
                # this contain a list
                # df = tabula.read_pdf(path_completo, pages = 1)
                # archivo_nuevo = archivo + '.csv'
                # path_completo_nuevo = os.path.join(directorio, archivo_nuevo)                
                # # Convert into Excel File
                # df.to_csv(path_completo_nuevo)

                # dfs = tabula.read_pdf(path_completo, pages='all')

                # for nth_frame, df in enumerate(dfs, start=1):
                #     csv_name = f'{path_completo}_{nth_frame}.csv'
                #     df.to_csv(csv_name, encoding='utf-8')

                # archivo_nuevo = archivo + '.csv'
                # path_completo_nuevo = os.path.join(directorio, archivo_nuevo)
                
                # tabula.convert_into(path_completo, path_completo_nuevo, output_format="csv", pages='all')

                # print(df)





                # reader=PdfReader(f) 
                # for page in reader.pages: 
                #     text+=page.extract_text()
                
                # data = pd.read_csv(StringIO(text), engine="python")
                # data.head()
                # print(data.head())



                # data = pd.read_csv("/content/drive/MyDrive/Content Creation/Youtube Tutorials/datasets/toxic_commnets_500.csv",error_bad_lines=False, engine="python")
                # data.head()






                # Para utilizar embeddings con Encoder 'multi-qa-MiniLM-L6-cos-v1' (Pendiente de funcionar)
                # reader=PdfReader(f) 
                # for page in reader.pages: 
                #     text+=page.extract_text()

                # model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
                # embeddings_new = model.encode(text)

                # print('embeddings_new: ')
                # print(embeddings_new)





                # Para utilizar con el modelo Word2Vec pre-entrenado (No funciona)
                # model = Word2Vec.load('assets/glove.840B.300d.txt')

                # # Tokeniza y procesa la consulta
                # query = "Qué es la embriogénesis?"
                # tokens = query.lower().split()

                # # Calcula el vector promedio de los tokens de la consulta
                # query_embedding = sum(model.wv[token] for token in tokens) / len(tokens)


                # query_results = pinecone.query(index_name, query_embedding, top_k=3)

                # print('query_results')
                # print(query_results)






                # Entrenar a la IA con el contenido separado de texts sin open ai: (Pendiente de funcionar)
                
                # index = pinecone.Index(index_name)

                # global qa
                
                # qa = qa.map(lambda x: {
                #     'encoding': model.encode(x['context']).tolist()
                # }, batched=True, batch_size=32)

                # upserts = [(v['id'], v['encoding']) for v in qa]

                # for i in tqdm(range(0, len(upserts), 50)):
                #     i_end = i + 50
                #     if i_end > len(upserts): i_end = len(upserts)
                #     index.upsert(vectors=upserts[i:i_end])

                # xq = model.encode([query]).tolist()

                # query = "Qué es la embriogénesis?"
                # xq = model.encode([query]).tolist()

                # xc = index.query(xq, top_k=5)

                # ids = [x['id'] for x in xc['results'][0]['matches']]

                # contexts = qa.filter(lambda x: True if x['id'] in ids else False)

                # print("contexts['context']")
                # print(contexts['context'])






                # Mover archivo entrenado (Paso Final)

                source_path = os.path.join(directorio, archivo)

                destination_path = os.path.join(directorio_destino, archivo)

                move_file(source_path, destination_path)

            except Exception as exc: 
                text=fallback_text_extraction(f)
                print('Exc: ' + exc)
                print('Text: ' + text)
                return JsonResponse({'success': False})

    return JsonResponse({'success': True})
    

def entrenar_ia_cohere(self):
    directorio = "assets/pdf_files_cohere"
    directorio_destino = "assets/pdf_files_cohere_procesados"
    collection_name = 'conversation_ai_1'
    # collection_name = request.json.get("collection_name")

    # text=""
    texts=""
    text=""

    # Recorrer los archivos del directorio
    for archivo in os.listdir(directorio):
        # Obtener el directorio de cada archivo
        path_completo= os.path.join(directorio, archivo)

        print(path_completo)
        print(archivo)

        loader = PyPDFLoader(path_completo)

        print(loader)

        docs = loader.load_and_split()

        print(docs)

        embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
        
        print(embeddings)

        qdrant = Qdrant.from_documents(docs, embeddings, url=qdrant_url, collection_name=collection_name, prefer_grpc=True, api_key=qdrant_api_key)
        
        source_path = os.path.join(directorio, archivo)

        destination_path = os.path.join(directorio_destino, archivo)

        move_file(source_path, destination_path)        
        # return {"collection_name":qdrant.collection_name}
    
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

# def gestionar_contenido(texts):
#     text_response_ai = ""
    
#     # Utilizar Pinecone para guardar los pdf nuevos y entrenar la IA:
#     docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
#     # docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)
#     print('docsearch')
#     print(docsearch)
#     # 'Qué es una organización?'
#     # 'Cómo es la embriogénesis?'
#     # 'Quién fue el padre de jesús?'
#     docs = docsearch.similarity_search('Qué es una organización?', include_metadata=True)
#     print('docs:')
#     print(docs)
    
#     # Conectar a la api de openAi
#     llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
#     chain = load_qa_chain(llm, chain_type="stuff")
    
#     text_response_ai = chain.run(input_documents=docs, question='Qué es una organización?')  

#     print(text_response_ai)  
#     return JsonResponse({'text_response_ai': text_response_ai}, safe=False)
 


def consultar_pdf(consulta):
    print('Consultar_pdf')
    print(consulta)
    print('list_indexes')
    print(pinecone.list_indexes())
    print(index_name)
    
    index = pinecone.GRPCIndex(index_name)

    print(index.describe_index_stats())

    text_field = "text"

    # switch back to normal index for langchain
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(
        index, embeddings.embed_query, text_field
    )

    query = consulta

    vector_result = vectorstore.similarity_search(
        query,  # our search query
        k=3  # return 3 most relevant docs
    )

    print('vector_result')
    print(vector_result)

    # print('Vector resultante: ')
    # print(vector_result[0])
    # text_response_ai = vector_result[0].page_content

    # Hacer pregunta al vector resultante a través de Open AI
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    text_response_ai = chain.run(input_documents=vector_result, question=query)
    print(text_response_ai)
    


#### Otra forma #################################################################################


    # Carga el modelo Word2Vec pre-entrenado
    # model = Word2Vec.load('ruta_al_modelo')

    # Tokeniza y procesa la consulta
    # query = "Mi consulta de ejemplo"
    # tokens = query.lower().split()

    # Calcula el vector promedio de los tokens de la consulta
    # query_embedding = sum(model.wv[token] for token in tokens) / len(tokens)


    # query_results = pinecone.query(index_name, query_embedding, top_k=3)

    # print('query_results')
    # print(query_results)


    # dataset = load_dataset('quora', split='train')
    # dataset

    # model = SentenceTransformer('bert-base-nli-mean-tokens')

    # model


#################################################################################################

    pusher_client.trigger('chat', 'message', {
        'theme': 'is-link',
        'username': 'Conversation UI',
        'message': text_response_ai
        })

    return JsonResponse({'success': True, 'text_response_ai': text_response_ai})

def consultar_pdf_cohere(consulta):
    print('Consultar_pdf')
    print(consulta)

    client = QdrantClient(url=qdrant_url, prefer_grpc=True, api_key=qdrant_api_key)

    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    # embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    qdrant = Qdrant(client=client, collection_name='conversation_ai_1', embeddings=embeddings)

    search_results = qdrant.similarity_search(consulta, k=2)

    # Hasta aquí no utiliza OPEN-AI
    print('Search_results')
    print(search_results)

    # Solo utiliza para Respuestas con lenguaje natural.

    chain = load_qa_chain(OpenAI(openai_api_key=OPENAI_API_KEY,temperature=0.2), chain_type="stuff")

    results = chain({"input_documents": search_results, "question": consulta}, return_only_outputs=True)

    # print(results["output_text"])

    # return {"results":results["output_text"]}

    # print(pinecone.list_indexes())
    # index = pinecone.GRPCIndex(index_name)

    # print(index.describe_index_stats())

    # text_field = "text"

    # switch back to normal index for langchain
    # index = pinecone.Index(index_name)

    # vectorstore = Pinecone(
    #     index, embeddings.embed_query, text_field
    # )

    # query = consulta

    # vector_result = vectorstore.similarity_search(
    #     query,  # our search query
    #     k=3  # return 3 most relevant docs
    # )

    # print('Vector resultante: ')
    # print(vector_result[0])
    # text_response_ai = vector_result[0].page_content

    # Hacer pregunta al vector resultante a través de Open AI
    # llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    # chain = load_qa_chain(llm, chain_type="stuff")
    # text_response_ai = chain.run(input_documents=vector_result, question=query)
    # print(text_response_ai)
    

    text_response_ai = results["output_text"]

    pusher_client.trigger('chat', 'message', {
        'theme': 'is-link',
        'username': 'Conversation UI',
        'message': text_response_ai
        })

    return JsonResponse({'success': True, 'text_response_ai': text_response_ai})


def upload_pdf(request):
    if request.method == 'POST':

        request.upload_handlers.pop(0)    

        print('upload_pdf')
        if request.method == 'POST' and request.FILES['pdf']:
            pdf_file = request.FILES['pdf']
            print('Nombre del PDF: ' + pdf_file.name)
            
            audio_file = request.FILES['audio_file']
            print('Nombre del audio: ' + audio_file.name)

            nombre_audio = audio_file.name + '.wav'

            print('nombre_audio: ' + nombre_audio)

            # Guardar archivo en el proyecto

            file_path = 'assets/audio_files/' + nombre_audio

            print('file_path: ' + file_path)
            
            with open(file_path, 'wb+') as archivo:
                for chunk in audio_file.chunks():
                    archivo.write(chunk)

            print("El archivo se ha guardado en:", file_path)
                     
            with open('assets/pdf_files/' + pdf_file.name, 'wb') as f:
                for chunk in pdf_file.chunks():
                    f.write(chunk)

            return read_pdf_sin_scan(pdf_file, file_path)
            # return JsonResponse({'success': True})
        else:
            return JsonResponse({'success': False})
        

def read_pdf_sin_scan(pdf_file, file_path):

    print('Dentro de read_pdf_sin_scan: ' + pdf_file.name)

    text=""

    try: 
        reader=PdfReader(pdf_file) 
        for page in reader.pages: 
            text+=page.extract_text()
 
        print('Try - Text: ' + text)

        return request_voice_prompt(text, file_path)

    except Exception as exc: 
        # text=fallback_text_extraction(pdf_file)
        print('Exc: ' + exc)
        print('Text: ' + text)

def request_voice_prompt(text, file_path):
    print('file_path dentro de request_voice_prompt: ' + file_path)
    # Obtener texto del audio Con Whisper desde la api de openAi: (Mucho más rápido)
    audio_file_2= open(file_path, "rb")
    
    transcript_2 = openai.Audio.transcribe("whisper-1", audio_file_2)

    print('transcript_2 dentro de request_voice_prompt:' + transcript_2["text"])

    text_prompt_2 = 'En relación al siguiente texto: "' + text +  '" la siguiente consulta: ' + transcript_2["text"]

    print('text_prompt_2 dentro de request_voice_prompt: ' + text_prompt_2)

    # Llamar a la API de OpenAI utilizando el texto generado por el audio como prompt
    model = "text-davinci-002"
    response = openai.Completion.create(
        engine=model,
        prompt=text_prompt_2,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Obtener el texto generado por el modelo de OpenAI
    generated_text_2 = response.choices[0].text
    print('generated_text_2 dentro de request_voice_prompt: ' + generated_text_2)

    # Responder a la interfaz gráfica la respuesta generada por openAi
    return JsonResponse({'status': 'Archivo de audio traducido a texto exitosamente.', 'generated_text_2': generated_text_2})


def upload_pdf_langchain(request):
    print(request)
    if request.method == 'POST':

        request.upload_handlers.pop(0)    

        print('upload_pdf_langchain')

        if request.method == 'POST' and request.FILES['pdf']:

            # Gestionar Audio:
            
            audio_file = request.FILES['audio_file']
            
            print('Nombre del audio: ' + audio_file.name)

            nombre_audio = audio_file.name + '.wav'

            print('nombre_audio: ' + nombre_audio)

            file_path_audio = 'assets/audio_files/' + nombre_audio

            print('file_path_audio: ' + file_path_audio)
            
            with open(file_path_audio, 'wb+') as archivo:
                for chunk in audio_file.chunks():
                    archivo.write(chunk)

            print("El archivo se ha guardado en:", file_path_audio)

            audio_file_to_transcript = open(file_path_audio, "rb")
    
            transcript = openai.Audio.transcribe("whisper-1", audio_file_to_transcript)

            transcript_text = transcript["text"]

            print('Contenido del audio dentro de upload_pdf_langchain: ' + transcript_text)

            # Gestionar Pdf:

            pdf_file = request.FILES['pdf']
            
            print('Nombre del PDF: ' + pdf_file.name)
            
            file_path = 'assets/pdf_files/' + pdf_file.name

            print('file_path: ' + file_path)
            
            with open(file_path, 'wb') as f:
                for chunk in pdf_file.chunks():
                    f.write(chunk)


            # Utilizar LangChain:

            loader = UnstructuredPDFLoader(file_path)
            data = loader.load()
            print (f'You have {len(data)} document(s) in your data')
            print (f'There are {len(data[0].page_content)} characters in your document')

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(data)
            print (f'Now you have {len(texts)} documents')


            # Utilizar Pinecone:
            print('index_name: ')
            print(index_name)
            docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
            
            # query = "What are examples of good data science teams?" (se volvió transcript_text)
            
            # docs = docsearch.similarity_search(transcript_text, include_metadata=True)

            # print('docs:')
            # print(docs)
            # print(docs[0].page_content[:250])

            # Conectar a la api de openAi
            llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
            chain = load_qa_chain(llm, chain_type="stuff")

            docs = docsearch.similarity_search(transcript_text, include_metadata=True)
            print('docs:')
            print(docs)

            # print(chain.run(input_documents=docs, question=transcript_text))
            text_response_ai = chain.run(input_documents=docs, question=transcript_text)
            # Retornar Respuesta al Front:
            return JsonResponse({'success': 'success', 'text_response_ai': text_response_ai})



@csrf_exempt
def send_pusher_event(request):
    print('request')
    # print(request.body)
    request.upload_handlers.pop(0)    
    data = json.loads(request.body)
    print(data)
    # print(data['consulta'])
    if request.method == 'POST':
        # print(data['consulta'])
        consultar_pdf(data['consulta'])
        # utilizar respuesta.consulta para consultar a pinecone

    return JsonResponse({'success': True})

@csrf_exempt

def send_pusher_event_cohere(request):
    request.upload_handlers.pop(0)    
    data = json.loads(request.body)
    print(data)
    # print(data['consulta'])
    if request.method == 'POST':
        consultar_pdf_cohere(data['consulta'])
        # utilizar respuesta.consulta para consultar a pinecone

    return JsonResponse({'success': True})
