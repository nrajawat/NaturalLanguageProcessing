import streamlit as st
# NLP Pkgs
from textblob import TextBlob
import spacy

# Function to Analyse Tokens and Lemma
def text_analyzer(Typed_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(Typed_text)
    # tokens = [ token.text for token in docx]
    allData = [('"Token":{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
    return allData

# Function For Extracting Entities
def entity_analyzer(Typed_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(Typed_text)
    tokens = [ token.text for token in docx]
    entities = [(entity.text,entity.label_)for entity in docx.ents]
    allData = ['"Tokens":{},\n"Entities":{}'.format(tokens,entities)]
    return allData

#main function
def main():

# Title
    st.title("Sentiment Analysis Application")
    st.subheader("Natural Language Processing")
    st.info("Streamlit & NLP Model")


# Tokenization
    if st.checkbox("Tokenization & Lemmatization"):
        st.subheader("Tokenize Your Text")

        message = st.text_area("Enter Text","Type Here ..")
        if st.button("Show Result"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

# Entity Extraction
    if st.checkbox("Recognize Named Entities"):
        st.subheader("Entities Extraction")
        message = st.text_area("Write something","Type..")
        if st.button("Extract Entities"):
            entity_result = entity_analyzer(message)
            st.json(entity_result)

# Sentiment Analysis
    if st.checkbox("Analyze Your Sentiments"):
        st.subheader("Sentiment Analysis")
        message = st.text_area("Write Something emotional","Type..")
        if st.button("Check Mood"):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)

            if result_sentiment.polarity >= 0.1:
                st.success("Good Mood")

            elif result_sentiment.polarity == 0.0:
                st.success("Peaceful")

            else:
                st.success("Bad Mood")


    st.sidebar.subheader("About App")
    st.sidebar.text("Sentiment Analysis App")
    st.sidebar.markdown(""" 
    #### Description  
    This is a Natural Language Processing(NLP) Application based on Streamlit. 
    It is useful for basic NLP task Tokenization,NER,Sentiment """)


    st.sidebar.subheader(" By")
    st.sidebar.text("Neelima Rajawat")
    st.sidebar.text("MS in Artificial Intelligence")
    st.sidebar.text("Florida Atlantic University")

if __name__ == '__main__':
    main()