import boto3
import json
import os
import botocore.config
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from pypdf import PdfReader, PdfMerger
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Loading environment variables
load_dotenv()

# Configure Bedrock client
boto3.setup_default_session(profile_name=os.getenv("profile_name"))
config = botocore.config.Config(connect_timeout=120, read_timeout=120)
bedrock = boto3.client('bedrock-runtime', 'us-east-1', endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                       config=config)


def summarizer(prompt_data) -> str:
    """
    This function creates the summary of each individual chunk as well as the final summary.
    :param prompt_data: This is the prompt along with the respective chunk of text, at the end it contains all summary chunks combined.
    :return: A summary of the respective chunk of data passed in or the final summary that is a summary of all summary chunks.
    """
    # Setting the key parameters to invoke Amazon Bedrock
    body = json.dumps({"prompt": prompt_data,
                       "max_tokens_to_sample": 8191,
                       "temperature": 0,
                       "top_k": 250,
                       "top_p": 0.5,
                       "stop_sequences": []
                       })
    # The specific Amazon Bedrock model you are using
    modelId = 'anthropic.claude-v2'
    #modelId = 'anthropic.claude-v2'
    # Type of data that should be expected upon invocation
    accept = 'application/json'
    contentType = 'application/json'
    # The invocation of Bedrock, with all of the parameters you have configured
    response = bedrock.invoke_model(body=body,
                                    modelId=modelId,
                                    accept=accept,
                                    contentType=contentType)
    # Gathering the response from Bedrock, and parsing to get specifically the answer
    response_body = json.loads(response.get('body').read())
    answer = response_body.get('completion')

    # Return the summary as plain text
    return answer

def num_tokens_from_string(string) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def create_pdf(text, filename):
    """
    Creates a PDF file with the given text.
    :param text: The text content to be included in the PDF.
    :param filename: The filename for the PDF file.
    """
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []
    paragraphs = text.split('\n\n')  # Split by double newlines to preserve paragraphs
    for paragraph in paragraphs:
        elements.append(Paragraph(paragraph, styles["BodyText"]))
    doc.build(elements)

def Chunk_and_Summarize(uploaded_files) -> str:
    """
    This function takes in a list of paths to the files that were just uploaded through the streamlit app.
    :param uploaded_files: A list of file paths, pointing to the newly uploaded files that are temporarily stored
    within the directory of this application.
    :return: This returns the final summary of the combined PDF documents that were initially passed in by the user through the
    streamlit app.
    """
    # Create a PdfMerger object
    merger = PdfMerger()

    # Concatenate all PDF files
    for pdf_file in uploaded_files:
        reader = PdfReader(pdf_file)
        merger.append(reader)

    # Create a new PDF file with all pages from the merged PDFs
    merged_pdf = open("merged.pdf", "wb")
    merger.write(merged_pdf)
    merged_pdf.close()

    # Read the merged PDF file as text
    reader = PdfReader("merged.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    # Creating the text splitter, using the recursive text splitter from langchain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )

    # Using the text splitter to split the entire text content of the merged PDF
    texts = text_splitter.create_documents([text])

    # Creating an empty summary string to append the summary of each chunk
    summary = ""

    # Looping through each chunk of text, generating a summary for each chunk
    for index, chunk in enumerate(texts):
        chunk_content = chunk.page_content
        prompt = f"""\n\nHuman: Provide a detailed summary for the chunk of text provided to you:
        Text: {chunk_content}
        \n\nAssistant:"""
        chunk_summary = summarizer(prompt)
        summary += chunk_summary + "\n\n"  # Add double newlines to separate chunk summaries
        print(f"\n\nChunk {index + 1} Summary:\n{chunk_summary}\n")
        print(f"Number of tokens for Chunk {index + 1} with the prompt: {num_tokens_from_string(prompt)} tokens")
        print("-------------------------------------------------------------------------------------------------------")

    # Creating the final summary prompt with all the individual chunk summaries
    final_summary_prompt = f"""\n\nHuman: You will be given a set of summaries from a document. Create a cohesive 
    summary from the provided individual summaries.  
    Summaries: {summary}
            \n\nAssistant:"""
    print(f"Number of tokens for this Chunk with the final prompt: {num_tokens_from_string(final_summary_prompt)}")

    # Generating the final summary of all the summaries
    final_summary = summarizer(final_summary_prompt)

    # Remove the temporary merged.pdf file
    os.remove("merged.pdf")

    # Create a PDF file with the final summary
    create_pdf(final_summary, "summary.pdf")

    # Return the final summary
    return final_summary