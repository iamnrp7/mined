import pandas as pd
import streamlit as st
import pdfplumber
import docx2txt
from bs4 import BeautifulSoup
import easyocr 
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain import globals as lc_globals
# from concurrent.futures import ThreadPoolExecutor

# Set verbose mode for langchain
lc_globals.set_verbose(True)

# Set page configuration
st.set_page_config(page_title="ResuMAGIC", page_icon="🌟", layout="wide")

# Load O*NET database (assuming it's in CSV format)
onet_database = "test_two.csv"
df_onet = pd.read_csv(onet_database)

# Initialize Gemini LLM with Google API key
google_api_key = "AIzaSyDX7iqE8XTN8npHp9jKZST8HMfZS4ncNpg"
llm = GoogleGenerativeAI(temperature=0.1, google_api_key=google_api_key, model="gemini-pro")

# Initialize Prompt Templates
first_input_prompt_template = "Please provide a rewritten version of {text}."
second_input_prompt_template = "Please extract and provide education details from {descript}."
work_prompt_template = "Please extract and provide work experience details from {text}."
projects_prompt_template = "Please extract and provide project details from {text}."
skills_prompt_template = "Please extract and provide skills from {text}."
career_trajectory_prompt_template = "Based on the provided {work_details} experience, analyze the career trajectory and give output in the following format: 1. job_title (start_date) >> job_title_2 (start_date) >> job_title_3 (start_date) >> job_title_4 (start_date) >> job_title_5 (start_date) >> job_title_6 (start_date). Please include only the start dates for each job title to analyze the career trajectory."

first_input_prompt = PromptTemplate(input_variables=['text'], template=first_input_prompt_template)
second_input_prompt = PromptTemplate(input_variables=['descript'], template=second_input_prompt_template)
work_prompt = PromptTemplate(input_variables=['text'], template=work_prompt_template)
projects_prompt = PromptTemplate(input_variables=['text'], template=projects_prompt_template)
skills_prompt = PromptTemplate(input_variables=['text'], template=skills_prompt_template)
career_trajectory_prompt = PromptTemplate(input_variables=['text'], template=career_trajectory_prompt_template)

# Chain of LLMs
chain1 = LLMChain(llm=llm, prompt=first_input_prompt, output_key='descript')
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, output_key='descript_two')

# Initialize Chains for Work, Projects, Skills, and Career Trajectory
work_chain = LLMChain(llm=llm, prompt=work_prompt, output_key='work_details')
projects_chain = LLMChain(llm=llm, prompt=projects_prompt, output_key='projects_details')
skills_chain = LLMChain(llm=llm, prompt=skills_prompt, output_key='skills_details')
career_trajectory_chain = LLMChain(llm=llm, prompt=career_trajectory_prompt, output_key='career_trajectory')

# Parent Chain for all tasks
parent_chain = SequentialChain(chains=[chain1, chain2, work_chain, projects_chain, skills_chain, career_trajectory_chain], input_variables=['text'], output_variables=['descript_two', 'work_details', 'projects_details', 'skills_details', 'career_trajectory'])

# Function to extract text from different file types
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == 'application/pdf':
        with pdfplumber.open(uploaded_file) as pdf:
            return '\n'.join([page.extract_text() for page in pdf.pages])
    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return docx2txt.process(uploaded_file)
    elif uploaded_file.type == 'text/html':
        soup = BeautifulSoup(uploaded_file, 'html.parser')
        return soup.get_text()
    elif uploaded_file.type.startswith('image/'):
        reader = easyocr.Reader(['en'])
        result = reader.readtext(uploaded_file.read())
        return '\n'.join([entry[1] for entry in result])

# Map job description to standardized job title
def map_job_description_to_title(job_description):
    similarity_scores = []
    for index, row in df_onet.iterrows():
        similarity_score = sum(a == b for a, b in zip(job_description.lower(), row['job_description'].lower())) / max(len(job_description), len(row['job_description']))
        similarity_scores.append((row['standerlised_tittle'], similarity_score))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[0][0] if sorted_scores[0][1] > 0 else None

# Streamlit UI
st.title('ResuMAGIC AI 🌟')

# File uploader for resume PDF, DOCX, HTML, and JPEG
uploaded_file = st.file_uploader("Upload Resume PDF, DOCX, HTML, or JPEG", type=['pdf', 'docx', 'html', 'jpeg', 'jpg'])

if uploaded_file is not None:
    with st.spinner("Analyzing..."):
        # Extract text from uploaded files
        extracted_text = extract_text_from_file(uploaded_file)
        if extracted_text:
            try:
                # With this line:
                result = parent_chain.invoke({'text': extracted_text})
               
                standardized_job_titles = [map_job_description_to_title(job_desc) for job_desc in result.get('descript_two', []) if job_desc]
                
                st.subheader("Check Out the Outcomes :")
                st.write(result.get('descript_two',[]))
                st.write(result.get('projects_details', []))
                st.write(result.get('skills_details', []))
                st.write("Career Trajectory:")
                st.write(result.get('career_trajectory', []))
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
else:
    st.info("Please upload a PDF, DOCX, HTML, or JPEG file to analyze.")
