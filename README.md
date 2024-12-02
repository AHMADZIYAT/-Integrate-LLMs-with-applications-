# -Integrate-LLMs-with-applications-
"""
author: Elena Lowery

This code sample shows how to invoke Large Language Models (LLMs) deployed in watsonx.ai.
Documentation: https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html
You will need to provide your IBM Cloud API key and a watonx.ai project id  (any project)
for accessing watsonx.ai in a .env file
This example shows simple use cases without comprehensive prompt tuning
"""

# Install the wml api your Python env prior to running this example:
# pip install ibm-watson-machine-learning
# pip install ibm-cloud-sdk-core

# In non-Anaconda Python environments, you may also need to install dotenv
# pip install python-dotenv

# For reading credentials from the .env file
import os
from dotenv import load_dotenv

# WML python SDK
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# For invocation of LLM with REST API
import requests, json
from ibm_cloud_sdk_core import IAMTokenManager

# Important: hardcoding the API key in Python code is not a best practice. We are using
# this approach for the ease of demo setup. In a production application these variables
# can be stored in an .env or a properties file

# URL of the hosted LLMs is hardcoded because at this time all LLMs share the same endpoint
url = "https://us-south.ml.cloud.ibm.com"

# These global variables will be updated in get_credentials() functions
watsonx_project_id = ""
# Replace with your IBM Cloud key
api_key = ""

def get_credentials():

    load_dotenv()

    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)

# The get_model function creates an LLM model object with the specified parameters
def get_model(model_type,max_tokens,min_tokens,decoding,temperature):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
        )

    return model

def get_list_of_complaints():

    # Look up parameters in documentation:
    # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#

    # You can specify any prompt and change parameters for different runs

    # If you want the end user to have a choice of the number of tokens in the output as well as decoding
    # and temperature, you can parameterize these values

    model_type = ModelTypes.LLAMA_2_13B_CHAT
    max_tokens = 100
    min_tokens = 50
    decoding = DecodingMethods.GREEDY
    # Temperature will be ignored if GREEDY is used
    temperature = 0.7

    # Instantiate the model
    model = get_model(model_type,max_tokens,min_tokens,decoding, temperature)

    complaint = f"""
            I just tried to book a flight on your incredibly slow website.  All 
            the times and prices were confusing.  I liked being able to compare 
            the amenities in economy with business class side by side.  But I 
            never got to reserve a seat because I didn't understand the seat map.  
            Next time, I'll use a travel agent!
            """

    # Hardcoding prompts in a script is not best practice. We are providing this code sample for simplicity of
    # understanding

    prompt_get_complaints = f"""
    From the following customer complaint, extract 3 factors that caused the customer to be unhappy. 
    Put each factor on a new line. 

    Customer complaint:{complaint}

    Numbered list of all the factors that caused the customer to be unhappy:

    """

    # Invoke the model and print the results
    generated_response = model.generate(prompt=prompt_get_complaints)
    # WML API returns a dictionary object. Generated response is a list object that contains generated text
    # as well as several other items such as token count and seed
    # We recommmend that you put a breakpoint on this line and example the result object
    print("---------------------------------------------------------------------------")
    print("Prompt: " + prompt_get_complaints)
    print("List of complaints: " + generated_response['results'][0]['generated_text'])
    print("---------------------------------------------------------------------------")

def answer_questions():

    # Look up parameters in documentation:
    # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#

    # You can specify any prompt and change parameters for different runs

    # If you want the end user to have a choice of the number of tokens in the output as well as decoding
    # and temperature, you can parameterize these values

    final_prompt = "Write a paragraph about the capital of France."
    model_type = ModelTypes.FLAN_UL2
    max_tokens = 300
    min_tokens = 50
    decoding = DecodingMethods.SAMPLE
    temperature = 0.7

    # Instantiate the model
    model = get_model(model_type,max_tokens,min_tokens,decoding, temperature)
    # Invoke the model and print the results
    generated_response = model.generate(prompt=final_prompt)
    # WML API returns a dictionary object. Generated response is a list object that contains generated text
    # as well as several other items such as token count and seed
    # We recommmend that you put a breakpoint on this line and example the result object
    print("---------------------------------------------------------------------------")
    print("Question/request: " + final_prompt)
    print("Answer: " + generated_response['results'][0]['generated_text'])
    print("---------------------------------------------------------------------------")

def invoke_with_REST():

    rest_url ="https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-29"

    access_token = get_auth_token()

    model_type = "google/flan-ul2"
    max_tokens = 300
    min_tokens = 50
    decoding = "sample"
    temperature = 0.7

    final_prompt = "Write a paragraph about the capital of France."

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer " + access_token
        }

    data = {
        "model_id": model_type,
        "input": final_prompt,
        "parameters": {
            "decoding_method": decoding,
            "max_new_tokens": max_tokens,
            "min_new_tokens": min_tokens,
            "temperature": temperature,
            "stop_sequences": ["."],
            },
        "project_id": watsonx_project_id
    }

    response = requests.post(rest_url, headers=headers, data=json.dumps(data))
    generated_response = response.json()['results'][0]['generated_text']

    print("--------------------------Invocation with REST-------------------------------------------")
    print("Question/request: " + final_prompt)
    print("Answer: " + generated_response)
    print("---------------------------------------------------------------------------")

def get_auth_token():

    # Access token is required for REST invocation of the LLM
    access_token = IAMTokenManager(apikey=api_key,url="https://iam.cloud.ibm.com/identity/token").get_token()
    return access_token

def demo_LLM_invocation():

    # Load the api key and project id
    get_credentials()

    # Show examples of 2 use cases/prompts
    answer_questions()
    get_list_of_complaints()

    # Simple prompt - invoked with the REST API
    invoke_with_REST()

demo_LLM_invocation()
"""
author: Elena Lowery

This code sample shows how to invoke Large Language Models (LLMs) deployed in watsonx.ai.
Documentation: # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
You will need to provide your IBM Cloud API key and a watonx.ai project id (any project)
for accessing watsonx.ai
This example shows a simple generation or Q&A use case without comprehensive prompt tuning
"""

# Install the wml and streamlit api your Python env prior to running this example:
# pip install ibm-watson-machine-learning
# pip install streamlit

# In non-Anaconda Python environments, you may also need to install dotenv
# pip install python-dotenv

# For reading credentials from the .env file
import os
from dotenv import load_dotenv

import streamlit as st

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# Important: hardcoding the API key in Python code is not a best practice. We are using
# this approach for the ease of demo setup. In a production application these variables
# can be stored in an .env or a properties file

# URL of the hosted LLMs is hardcoded because at this time all LLMs share the same endpoint
url = "https://us-south.ml.cloud.ibm.com"

# These global variables will be updated in get_credentials() functions
watsonx_project_id = ""
# Replace with your IBM Cloud key
api_key = ""

def get_credentials():

    load_dotenv()

    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)

    print("*** Got credentials***")

# The get_model function creates an LLM model object with the specified parameters
def get_model(model_type,max_tokens,min_tokens,decoding,stop_sequences):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.STOP_SEQUENCES:stop_sequences
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
        )

    return model

def get_prompt(question):

    # Prompts are passed to LLMs as one string. We are building it out as separate strings for ease of understanding
    # Instruction
    instruction = "Answer this question briefly."
    # Examples to help the model set the context
    examples = "\n\nQuestion: What is the capital of Germany\nAnswer: Berlin\n\nQuestion: What year was George Washington born?\nAnswer: 1732\n\nQuestion: What are the main micro nutrients in food?\nAnswer: Protein, carbohydrates, and fat\n\nQuestion: What language is spoken in Brazil?\nAnswer: Portuguese \n\nQuestion: "
    # Question entered in the UI
    your_prompt = question
    # Since LLMs want to "complete a document", we're are giving it a "pattern to complete" - provide the answer
    end_prompt = "Answer:"

    final_prompt = instruction + examples + your_prompt + end_prompt

    return final_prompt

def answer_questions():

    # Set the api key and project id global variables
    get_credentials()

    # Web app UI - title and input box for the question
    st.title('🌠Test watsonx.ai LLM')
    user_question = st.text_input('Ask a question, for example: What is IBM?')

    # If the quesiton is blank, let's prevent LLM from showing a random fact, so we will ask a question
    if len(user_question.strip())==0:
        user_question="What is IBM?"

    # Get the prompt
    final_prompt = get_prompt(user_question)

    # Display our complete prompt - for debugging/understanding
    print(final_prompt)

    # Look up parameters in documentation:
    # https://ibm.github.io/watson-machine-learning-sdk/foundation_models.html#
    model_type = ModelTypes.FLAN_UL2
    max_tokens = 100
    min_tokens = 20
    decoding = DecodingMethods.GREEDY
    stop_sequences = ['.']

    # Get the model
    model = get_model(model_type, max_tokens, min_tokens, decoding,stop_sequences)

    # Generate response
    generated_response = model.generate(prompt=final_prompt)
    model_output = generated_response['results'][0]['generated_text']
    # For debugging
    print("Answer: " + model_output)

    # Display output on the Web page
    formatted_output = f"""
        **Answer to your question:** {user_question} \
        *{model_output}*</i>
        """
    st.markdown(formatted_output, unsafe_allow_html=True)

# Invoke the main function
answer_questions()
python3 -m pip install -r requirements.txt


Analisis 

1. Fitur Utama LLM 🌐

Bayangkan LLM sebagai superkomputer cerdas yang menguasai bahasa dan kreativitas.

💡 Pemahaman Kontekstual yang Tajam:

LLM dapat membaca antara baris-baris teks, memahami makna mendalam, dan membuat koneksi logis. Ini seperti memiliki teman yang selalu memahami maksud Anda tanpa penjelasan panjang lebar.

✨ Fleksibilitas Tanpa Batas:

Dari penulisan skrip hingga menjawab pertanyaan teknis, LLM adalah asisten multitasking untuk semua kebutuhan Anda.
🔧 Kemampuan Penyempurnaan:
Melatih LLM ulang untuk keperluan spesifik (fine-tuning) memungkinkan model bekerja lebih efektif pada domain tertentu—misalnya, untuk bahasa lokal atau istilah teknis industri.

📊 Visualisasi GitHub:

Tambahkan GIF yang menggambarkan proses LLM memahami teks, seperti buku-buku yang melayang ke dalam model dan keluar sebagai jawaban.

Gunakan ikon interaktif (🌐, 🔧) untuk menunjukkan fitur utama.

2. Dasar-Dasar Rekayasa Cepat (Prompt Engineering) 🎯
Berinteraksi dengan LLM seperti menyusun pertanyaan ajaib yang mengarahkan ke jawaban terbaik.

📜 Prinsip Merancang Prompt:

Prompt yang jelas = jawaban yang tepat. Misalnya, “Tuliskan ringkasan artikel” lebih efektif dibandingkan “Buat teks singkat.”

⚙️ Penyetelan Parameter:

Kendalikan kreativitas LLM dengan parameter:

🔥 Temperatur tinggi → Hasil lebih kreatif (untuk cerita atau puisi).

❄️ Temperatur rendah → Hasil lebih faktual (untuk laporan teknis).

🔄 Eksperimen Iteratif:

Jangan takut mencoba berbagai format prompt untuk menemukan yang paling efektif.

📊 Visualisasi GitHub:

Tambahkan diagram animasi yang menunjukkan efek perubahan suhu (temperature) pada keluaran LLM.
Tampilkan kode contoh dengan efek perubahan warna untuk membedakan prompt yang efektif dan tidak.

3. Menggunakan Prompt Lab di Watsonx.ai 🧪
   
Prompt Lab adalah laboratorium kreatif untuk memaksimalkan potensi LLM.

🖥️ Antarmuka yang Intuitif:

Alat ini memudahkan eksplorasi berbagai jenis prompt tanpa memerlukan pengalaman teknis mendalam.

⚡ Pengujian Model yang Cepat:

Bandingkan respons dari beberapa model secara paralel, seperti memilih pemain terbaik untuk tim Anda.

🌈 Fleksibilitas Eksperimen:

Cocok untuk skenario formal hingga kasual. Misalnya, membuat email profesional atau menulis caption media sosial.

📊 Visualisasi GitHub:

Tambahkan video demo singkat yang menunjukkan Prompt Lab di watsonx.ai dengan fokus pada perubahan prompt dan hasilnya.
Gunakan ikon model paralel untuk menyoroti perbandingan hasil.
4. Menguji Inferensi Model LLM ⚙️
Menguji LLM seperti mengetes mesin pintar—seberapa cepat dan akurat ia merespons.

🚀 Kecepatan Inferensi:

Tergantung ukuran model dan sumber daya komputasi. Model besar cenderung lebih lambat tetapi lebih akurat.

🎯 Akurasi Respons:

Keluaran bergantung pada kualitas prompt dan kemampuan generalisasi model.

🔍 Analisis Respons:

Memahami logika model membantu meningkatkan efektivitas interaksi.

📊 Visualisasi GitHub:

Tambahkan grafik interaktif kecepatan dan akurasi inferensi berdasarkan parameter yang diubah.
Gunakan kode blok dinamis untuk menampilkan prompt dan hasil yang diperoleh.

5. Membuat UI Sederhana untuk Interaksi LLM 🖌️
   
Membuat UI untuk LLM seperti membangun jembatan antara manusia dan mesin pintar.

🔗 Integrasi Model:

Hubungkan API watsonx.ai ke antarmuka menggunakan framework seperti Flask atau React.

🎨 Desain UI/UX:

Buat antarmuka intuitif, dengan area input prompt, pengaturan parameter, dan hasil respons.

🛠️ Fitur Utama:

Tambahkan slider untuk mengatur suhu (temperature), panjang output, dan opsi pemilihan model.

📊 Visualisasi GitHub:

Tambahkan tangkapan layar atau demo interaktif antarmuka yang Anda buat.
Gunakan ikon UI/UX untuk menonjolkan elemen fungsional antarmuka.
