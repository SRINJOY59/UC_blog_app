import os
import pdfkit
import pathlib
import textwrap
import requests
import streamlit as st
from io import BytesIO
import base64
from bs4 import BeautifulSoup
import google.generativeai as genai
from IPython.display import display, Markdown
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
warnings.filterwarnings("ignore")

os.environ['GOOGLE_API_KEY'] = 'AIzaSyCTjKjVn1IlY1e9hwMh7NzWmWsHXSO9KoI'

def to_markdown(text):
    text = text.replace('.', '*')
    return textwrap.indent(text, '>', predicate=lambda _: True)

import pandas as pd

df = pd.read_json('data.json')


def extract_relevant_documents(pdf_path, query):
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
    context = "\n\n".join(str(page.page_content) for page in pages)
    texts = text_splitter.split_text(context)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

    relevant_documents = vector_index.get_relevant_documents(query)

    return relevant_documents


llm = ChatGoogleGenerativeAI(model="gemini-pro")

def generate_blog(topic,category,knowledge=None):
    header_prompt = """You are a blog writer for a reputed organization writing blogs on {category}. Suggest a SEO optimized long heading for a blog on {name}. The heading
                   must be of \'14 words\' to make a complete sentence. Make limited use of positive sentiment and use /'context words'/ to capture attention and
                   build trust. Talk about the human body as per {name}. Do not use \'passive language\'such as \'chance'\, \'doubt\', \'confuse\',
                   \'guess\'. Use more \'alert words\' such as \'afraid\', \'scare\', \'risk\', \'alarm\'. Also try to humanize the heading"""
    blog_prompt = """You are a senior blog writer for an organization, where your task is to write SEO optimized and SERP analyzed informational blogs within the category of {category}.
                          Your task is to {knowledge}write a proper blog of \'1500 words\' on the topic of {topic}. Use the following example of a blog on {eg_topic} to get an idea,
                          with the following content: {eg_content}. Use h2 and h3 headers. /*Do not use bullet points or numbered points*/, just the headings:\n
                          {heading}"""
    knowledge_base_prompt = "Gather all details from the following context and "

    header_prompt = header_prompt.format(category=category, name=topic)

    heading = llm.invoke(header_prompt).content

    link = df[category]['links'][0]
    res = requests.get(link)

    soup = BeautifulSoup(res.text,'html.parser')
    eg_content = soup.find('div', class_='post-details').text
    eg_topic = df[category]['titles'][0]

    blog_prompt = blog_prompt.format(category=category,
                                                   topic=topic,
                                                   heading=heading,
                                                   eg_topic=eg_topic,
                                                   eg_content=eg_content,
                                                   knowledge=knowledge)
    blog = llm.invoke(blog_prompt).content
    blog = '\n'.join(blog.split('\n')[1:])
    blog = '# '+heading + '\n' + blog
    return blog


def write_blog(pdf_path, prompt, cat):
    docs = extract_relevant_documents(pdf_path, prompt)
    blog = generate_blog(topic=prompt,
           category=cat, knowledge = docs)
    return to_markdown(blog)


import streamlit as st

st.title('Blog Generator from PDF Content')

# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")


input_text = st.text_area("Enter your text here", height=300)

prompt = st.text_input("Enter the blog topic")

category = st.selectbox("Choose the blog category", ['beauty', 'happy-homes', 'interiors', 'wellness', 'lifestyle', 'uc impact'])

if st.button('Generate Blog'):
    # if uploaded_file is not None and prompt:
    #     pdf_path = "uploaded_pdf.pdf"  
    #     with open(pdf_path, "wb") as f:
    #         f.write(uploaded_file.read())
    #     blog_markdown = write_blog(pdf_path, prompt, category)
        
    #     st.markdown(blog_markdown, unsafe_allow_html=True)
    if input_text is not None and prompt:
        blog_response = generate_blog(prompt, category, input_text)
        st.markdown(blog_response, unsafe_allow_html=True)
    else:
        st.error("Please upload a PDF file and enter a prompt.")
