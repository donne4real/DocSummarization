
import boto3
import json
import os
import botocore.config
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import tiktoken
from pypdf import PdfReader
from fpdf import FPDF
from textwrap import wrap

# loading environment variables
load_dotenv()
# configure Bedrock client
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
    # setting the key parameters to invoke Amazon Bedrock
    body = json.dumps({"prompt": prompt_data,
                       "max_tokens_to_sample": 8191,
                       "temperature": 0,
                       "top_k": 250,
                       "top_p": 0.5,
                       "stop_sequences": []
                       })
    # the specific Amazon Bedrock model you are using
    modelId = 'anthropic.claude-v2'
    # type of data that should be expected upon invocation
    accept = 'application/json'
    contentType = 'application/json'
    # the invocation of bedrock, with all of the parameters you have configured
    response = bedrock.invoke_model(body=body,
                                    modelId=modelId,
                                    accept=accept,
                                    contentType=contentType)
    # gathering the response from bedrock, and parsing to get specifically the answer
    response_body = json.loads(response.get('body').read())
    answer = response_body.get('completion')
    # returning the final summary for that chunk of text
    return answer


def num_tokens_from_string(string) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def Chunk_and_Summarize(uploaded_files) -> None:
    """
    This function takes in the paths to the files that were just uploaded through the streamlit app.
    :param uploaded_files: A list of file paths, that should point to the newly uploaded files that are temporarily stored
    within the directory of this application.
    :return: None. This function generates and saves PDF summary files for each uploaded file.
    """
    for uploaded_file in uploaded_files:
        # using PyPDF PdfReader to read in the PDF file as text
        reader = PdfReader(uploaded_file)
        # creating an empty string for us to append all the text extracted from the PDF
        text = ""
        # a simple for loop to iterate through all pages of the PDF we uploaded
        for page in reader.pages:
            # as we loop through each page, we extract the text from the page and append it to the "text" string
            text += page.extract_text() + "\n"
        # creating the text splitter, we are specifically using the the recursive text splitter from langchain:
        # https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True
        )
        # using the text splitter to split the entire string of text that contains all the text content of our PDF
        texts = text_splitter.create_documents([text])
        # Creating an empty summary string, as this is where we will append the summary of each chunk
        summary = ""
        # looping through each chunk of text we created, passing that into our prompt and generating a summary of that chunk
        for index, chunk in enumerate(texts):
            # gathering the text content of that specific chunk
            chunk_content = chunk.page_content
            # creating the prompt that will be passed into Bedrock with the text content of the chunk
            prompt = f"""\n\nHuman: Provide a detailed summary for the chunk of text provided to you:
            Text: {chunk_content}
            \n\nAssistant:"""
            # passing the prompt into the summarizer function to generate the summary of that chunk, and appending it to
            # the summary string
            summary += summarizer(prompt)
            # printing out the number of tokens contained in each chunk to provide a status update
            print(f"\n\nNumber of tokens for Chunk {index + 1} with the prompt: {num_tokens_from_string(prompt)} tokens")
            print("-------------------------------------------------------------------------------------------------------")
        # after we have generated the summaries of each chunk of text, and appended them to the single summary string,
        # we pass it into the final summary prompt
        final_summary_prompt = f"""\n\nHuman: You will be given a set of summaries from a document. Create a cohesive 
        summary from the provided individual summaries. The summary should very detailed and at least 10 pages. 
        Summaries: {summary}
                \n\nAssistant:"""
        # print the total number of tokens being passed into the final summarization prompt
        print(f"Number of tokens for this Chunk with the final prompt: {num_tokens_from_string(final_summary_prompt)}")

        # generating the final summary of all the summaries we have previously generated.
        final_summary = summarizer(final_summary_prompt)

        # Create a PDF file from the final summary
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        page_width = pdf.w - 2 * pdf.l_margin

        line_height = pdf.font_size * 1.2
        y_pos = pdf.y + line_height
        for line in final_summary.split('\n'):
            words = line.strip().split()
            while words:
                current_line = ''
                while words and pdf.get_string_width(current_line + words[0]) <= page_width:
                    current_line += words.pop(0) + ' '
                pdf.cell(page_width, line_height, current_line.strip(), align='L', ln=2)
                y_pos += line_height

        # Save the PDF file with the uploaded file name
        pdf_filename = os.path.splitext(os.path.basename(uploaded_file))[0] + "_summary.pdf"
        pdf.output(pdf_filename)
        print(f"Summary PDF file saved: {pdf_filename}")


# Example usage
uploaded_files = ['path/to/file1.pdf', 'path/to/file2.pdf', 'path/to/file3.pdf']
Chunk_and_Summarize(uploaded_files)
