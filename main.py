import streamlit as st
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
import chromadb

st.title('AI-Powered Project Risk Management System')

# Initialize ChromaDB client
db = chromadb.Client()

# Set up the OpenAI LLM
llm = OpenAI(temperature=0.5)

# Define a simple prompt template
prompt = PromptTemplate(input_variables=['project_data'], template='Based on the following project data, identify potential risks: {project_data}')

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

st.sidebar.header('Input Data')
project_data = st.sidebar.text_area('Project Data', 'Enter your project details here...')

if st.sidebar.button('Analyze Risks'):
    if project_data:
        response = chain.run(project_data=project_data)
        st.subheader('Identified Risks')
        st.write(response)
    else:
        st.warning('Please enter project data to analyze.')