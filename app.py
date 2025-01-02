import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document

# Load environment variables
load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")

# Initialize ChatGroq LLM
llm = ChatGroq(groq_api_key=groq_api, model='gemma2-9b-it')

# Set up the summarization prompt
genericprompt = """
Write a summary of the following text in points with a Heading:
Text: {text}
Translate the precise summary to {language}
"""

prompt = PromptTemplate(
    input_variables=['text', 'language'],
    template=genericprompt
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit App
st.title("Text Summarization")

# Sidebar with options
st.sidebar.header("Summarization Options")
technique = st.sidebar.radio(
    "Choose one:",
    ("Input Text", "Upload PDF", "Web URL","Youtube Video")
)

# Language selection


# Handle Input Techniques
if technique == "Input Text":
    language = st.selectbox("Choose the translation language for the summary:", ("English", "Hindi", "Spanish", "French"))
    
    text = st.text_area("Enter the text you want to summarize:", height=200)

    # Summarize Button
    if st.button("Summarize"):
        if text.strip():
            try:
                summary = llm_chain.run({"text": text, "language": language})
                st.subheader("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to summarize.")

elif technique == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file:", type="pdf")

    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(uploaded_file.read())
                temp_pdf_path = temp_pdf.name

            st.info("Loading the PDF. Please wait...")

            # Use PyPDFLoader to load and split the document
            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()

            # Use text splitter for efficient chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            docs = text_splitter.split_documents(documents)

            # Display status
            st.success(f"Successfully loaded and split PDF into {len(docs)} chunks.")

            # Prompts for summarization
            chunks_prompt = """ 
            Please summarize the below text:
            Text: {text}
            Summary: 
            """
            map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)
            
            final_prompt = """ 
            Provide the final summary of the entire document with these important points.
            Add a Strong Title, Start the summary with an introduction and provide the summary in numbered points.
            Text: {text}
            """
            final_prompt_template = PromptTemplate(input_variables=['text'], template=final_prompt)

            # Summarization chain using map-reduce
            summary_chain = load_summarize_chain(
                llm=llm,
                chain_type='map_reduce',
                map_prompt=map_prompt_template,
                combine_prompt=final_prompt_template,
                verbose=True
            )
            
            if st.button("Summarize PDF"):
                st.info("Generating summary. Using Map and Reduce Summarization Technique.. This may take some time...")
                output = summary_chain.run(docs)
                st.subheader("PDF Summary:")
                st.write(output)
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")

elif technique == "Web URL":
    url = st.text_input("Enter the Web URL:")

    if url.strip():
        if st.button("Summarize URL"):
            try:
                loader = WebBaseLoader(url)
                documents = loader.load_and_split()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                docs = text_splitter.split_documents(documents)

                
                summary_chain = load_summarize_chain(llm=llm,chain_type='refine',verbose=True)
                output = summary_chain.run(docs)
                
                summary = output
                st.subheader("Web URL Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred while processing the URL: {e}")
        else:
            st.warning("Using Refine Summarization Technique")

elif technique == "Youtube Video":
    url = st.text_input("Enter URL for Youtube Video..")

    if url.strip():
        if st.button("Summarize URL"):
            try:
                # Extract video ID from the URL
                video_id = url.split("v=")[-1]

                # Get the transcript of the YouTube video
                st.info("Fetching transcript from YouTube...")
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                video_text = " ".join([entry['text'] for entry in transcript])

                # Create a Document object
                document = Document(page_content=video_text, metadata={"source": "YouTube Video"})

                # Use RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                docs = text_splitter.split_documents([document])  # Pass the Document in a list

                # Summarization using refine chain
                summary_chain = load_summarize_chain(llm=llm, chain_type='refine', verbose=True)
                output = summary_chain.run(docs)

                # Display the summary
                st.subheader("YouTube Video Summary:")
                st.write(output)
            except Exception as e:
                st.error(f"An error occurred while processing the URL: {e}")
        else:
            st.warning("Using Refine Summarization Technique")
