from django.shortcuts import render
from django.http import JsonResponse
# from django.http import HttpResponse
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


import openai
# import whisper
# import io


# import pytesseract
# import PyPDF2
# from PIL import Image
from PyPDF2 import PdfReader 
from pdfminer.high_level import extract_text as fallback_text_extraction 


# Create your views here.


openai.api_key = "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"



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



def upload_pdf(request):
    if request.method == 'POST':

        request.upload_handlers.pop(0)    

        print('upload_pdf')
        if request.method == 'POST' and request.FILES['pdf']:
            pdf_file = request.FILES['pdf']
            print(pdf_file.name)
            with open('assets/audio_files/' + pdf_file.name, 'wb') as f:
                for chunk in pdf_file.chunks():
                    f.write(chunk)
            read_pdf_sin_scan(pdf_file)
            # read_pdf_con_scan(pdf_file)
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'success': False})
        


# def read_pdf_con_scan(pdf):
#     print('Dentro de read_pdf_con_scan: ' + pdf.name)



def read_pdf_sin_scan(pdf):

    print('Dentro de read_pdf_sin_scan: ' + pdf.name)

    text=""

    try: 
        reader=PdfReader(pdf) 
        for page in reader.pages: 
            text+=page.extract_text()
 
        print('Try - Text: ' + text)

        request_voice_prompt(text)

    except Exception as exc: 
        text=fallback_text_extraction(pdf)
        print('Exc: ' + exc)
        print('Text: ' + text)

def request_voice_prompt(text):

    text_prompt = 'En relación al siguiente texto: "' + text +  '" la siguiente consulta: ' 

    print(text_prompt)