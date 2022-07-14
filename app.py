from html import entities
import streamlit as st 
import streamlit.components.v1 as stc

# Text Cleaning Pkgs
import neattext as nt
import neattext.functions as nfx

# utils
from collections import Counter
import time
import base64

# Text Viz Pkgs
from wordcloud import WordCloud 
from textblob import TextBlob

# Load NLP pkgs
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy import displacy

# File Processing Pkgs
import docx2txt
import pdfplumber




# Data EDA Pkgs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib 
matplotlib.use('Agg')
import altair as alt 
from PIL import Image
import os
from pathlib import Path

#Func to read pdf
from PyPDF2 import PdfFileReader
import pdfplumber

# FUNCTIONS
def text_analyzer(my_text):
    docx = nlp(my_text)
    allData = [(token.text,token.shape_,token.pos_,token.tag_,token.lemma_,token.is_alpha,token.is_stop) for token in docx]
    
    df = pd.DataFrame(allData ,columns=['Tokens','Shape','POS','Tag','Lemma','IsAlpha','Is Stop_word'])
    return df
   

def get_entities(my_text):
    docx = nlp(my_text)
    entities = [(entity.text.entity.label_) for entity in docx.ents]
    return entities


HTML_WRAPPER = """
    <div style="overflow-x: auto; border: 1px solid #e6e9ef;border-radius:5px;border-style:ridge;margin:5px;">
    </div>
    """
#   <h1 style="color:white;text-align:center;">Text Analyzer </h1>
def render_entities(rawtext):
    docx = nlp(rawtext)
    html = displacy.render(docx,style='ent')
    html = html.replace("\n\n","\n")
    result = HTML_WRAPPER.format(html)
    
    return result

def get_most_common_tokens(docx,num=10):
	word_freq = Counter(docx.split())
	most_common_tokens = word_freq.most_common(num)
	return dict(most_common_tokens)

#Function to Get Sentiment
def get_sentiment(my_text):
    blob = TextBlob(my_text)
    sentiment = blob.sentiment
    return sentiment


def plot_wordcloud(docx):
	mywordcloud = WordCloud().generate(docx)
	fig = plt.figure(figsize=(20,10))
	plt.imshow(mywordcloud,interpolation='bilinear')
	plt.axis('off')
	st.pyplot(fig)



def read_pdf(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text == page.extractText()

    return all_page_text

def read_pdf2(file):
    with pdfplumber.open(file) as pdf:
        page = pdf.pages[0]
        return page.extract_text()





def main():
    st.title("Text Analyzer")
    menu = ["Home","Upload Files","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == 'Home':
        st.subheader("Text Analysis")
        raw_text = st.text_area('Enter Text Here')
        num_of_most_common = st.sidebar.number_input("Most Common Tokens",5,15)


        if st.button("Analyze"):
            if len(raw_text) > 2:
                with st.expander("Original Text"):
                    st.write(raw_text)
                
                with st.expander("Text Analysis"):
                    token_results_df = text_analyzer(raw_text)
                    st.dataframe(token_results_df)
                
                # with st.expander("Entities"):
                #     entity_result = render_entities(raw_text)
                #     stc.html(entity_result,height=1000,scrolling=True)

                # Layouts
                col1,col2 = st.columns(2)

                with col1:
                    with st.expander("Word Stats"):
                        st.info("Word Statistics")
                        docx = nt.TextFrame(raw_text)
                        st.write(docx.word_stats())

                    with st.expander("Top Keywords"):
                        st.info("Top Keywords/Tokens")
                        processed_text = nfx.remove_stopwords(raw_text)
                        keywords = get_most_common_tokens(processed_text,num_of_most_common)
                        st.write(keywords)


                    with st.expander("Sentiment"):
                        sent_result = get_sentiment(raw_text)
                        st.write(sent_result)
                        
                
                with col2:
                    with st.expander("Plot Word Freq"):
                        fig = plt.figure()
                        top_keywords = get_most_common_tokens(processed_text,num_of_most_common )
                        plt.bar(keywords.keys(),top_keywords.values())
                        plt.xticks(rotation = 45)
                        st.pyplot(fig)



                    with st.expander("Plot Part of Speech(POS)"):
                        fig = plt.figure()
                        sns.countplot(token_results_df['POS'])
                        plt.xticks(rotation = 45)
                        st.pyplot(fig)

                    with st.expander("Plot Wordcloud"):
                        plot_wordcloud(raw_text)
                

            # with st.expander("Download Text Analysis Results"):
            #     pass
            



    elif choice == "Upload Files":
        st.subheader("Upload your own file (pdf,txt)")
        
        text_file = st.file_uploader("Upload Files",type=['pdf','docx','txt'])
        num_of_most_common = st.sidebar.number_input("Most Common Tokens",5,15)
        if text_file is not None:
            if text_file.type == "application/pdf":
                raw_text = read_pdf(text_file)
                # st.write(raw_text)
            elif text_file.type == 'text/plain':
                raw_txt = str(text_file.read(),"utf-8")
                # st.write(raw_text)
            else:
                raw_text = docx2txt.process(text_file)
                # st.write(raw_text)


            # with st.expander("Original Text"):
            #     # st.write(raw_text)
            
            with st.expander("Text Analysis"):
                token_results_df = text_analyzer(raw_text)
                st.dataframe(token_results_df)
            
            with st.expander("Entities"):
                entity_result = render_entities(raw_text)
                stc.html(entity_result,height=1000,scrolling=True)

                # Layouts
            col1,col2 = st.columns(2)

            with col1:
                with st.expander("Word Stats"):
                    st.info("Word Statistics")
                    docx = nt.TextFrame(raw_text)
                    st.write(docx.word_stats())

                with st.expander("Top Keywords"):
                    st.info("Top Keywords/Tokens")
                    processed_text = nfx.remove_stopwords(raw_text)
                    keywords = get_most_common_tokens(processed_text,num_of_most_common)
                    st.write(keywords)


                with st.expander("Sentiment"):
                    sent_result = get_sentiment(raw_text)
                    st.write(sent_result)
                    
            
            with col2:
                with st.expander("Plot Word Freq"):
                    fig = plt.figure()
                    top_keywords = get_most_common_tokens   (processed_text,num_of_most_common )
                    plt.bar(keywords.keys(),top_keywords.values())
                    plt.xticks(rotation = 45)
                    st.pyplot(fig)



                with st.expander("Plot Part of Speech(POS)"):
                    fig = plt.figure()
                    sns.countplot(token_results_df['POS'])
                    plt.xticks(rotation = 45)
                    st.pyplot(fig)

                with st.expander("Plot Wordcloud"):
                    plot_wordcloud(raw_text)
                
                

 
        



if __name__ == '__main__':
	main()