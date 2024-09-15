"""
URL configuration for conversation_ai_api project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from chatbot.views import conversation
from chatbot.views import guardar_archivo_audio
from chatbot.views import upload_pdf
from chatbot.views import subir_pdf
# from chatbot.views import subir_pdf_cohere
from chatbot.views import entrenar_ia
# from chatbot.views import entrenar_ia_cohere
from chatbot.views import consultar_pdf
# from chatbot.views import consultar_pdf_cohere
from chatbot.views import upload_pdf_langchain
from chatbot.views import send_pusher_event
# from chatbot.views import send_pusher_event_cohere
from django.views.decorators.csrf import csrf_exempt
from chatbot.views import obtener_listado_archivos
# from chatbot.views import obtener_listado_archivos_cohere
from chatbot.views import obtener_listado_archivos_procesados
# from chatbot.views import obtener_listado_archivos_cohere_procesados

urlpatterns = [
    path('admin/', admin.site.urls),
    path('conversation/', conversation),
    path('entrenar_ia/', entrenar_ia),
    # path('entrenar_ia_cohere/', entrenar_ia_cohere),
    path('consultar_pdf/', consultar_pdf),
    # path('consultar_pdf_cohere/', consultar_pdf_cohere),
    path('obtener_listado_archivos/', obtener_listado_archivos),
    # path('obtener_listado_archivos_cohere/', obtener_listado_archivos_cohere),
    path('obtener_listado_archivos_procesados/', obtener_listado_archivos_procesados),
    # path('obtener_listado_archivos_cohere_procesados/', obtener_listado_archivos_cohere_procesados),
    path('send_pusher_event/', send_pusher_event),
    # path('send_pusher_event_cohere/', send_pusher_event_cohere),
    path('audio/', csrf_exempt(guardar_archivo_audio), name='guardar_archivo_audio'),
    path('upload_pdf/', csrf_exempt(upload_pdf), name='upload_pdf'),
    path('subir_pdf/', csrf_exempt(subir_pdf), name='subir_pdf'),
    # path('subir_pdf_cohere/', csrf_exempt(subir_pdf_cohere), name='subir_pdf_cohere'),
    
    path('upload_pdf_langchain/', csrf_exempt(upload_pdf_langchain), name='upload_pdf_langchain'),
]