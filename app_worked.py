import streamlit as st
from pathlib import Path
import sys
from os import path
from doc_summarizer import Chunk_and_Summarize
import time
from dotenv import load_dotenv
from tempfile import TemporaryDirectory

# load environment variables
load_dotenv()
# title of the streamlit app
st.title(f""":rainbow[Long Document Summarization with Amazon Bedrock]""")

# default container that houses the document upload field
with st.container():
    # header that is shown on the web UI
    st.header('Multiple File Upload')
    # the file upload field, the specific ui element that allows you to upload the file
    Files = st.file_uploader('Upload files', type=["pdf"], accept_multiple_files=True, key="new")
    # when files are uploaded, save them to a temporary directory, create paths, and invoke the
    # Chunk_and_Summarize Function
    if Files:
        # create a temporary directory to save the uploaded files
        with TemporaryDirectory() as temp_dir:
            # create a list of save paths for each uploaded file
            save_paths = [Path(temp_dir, file.name) for file in Files]

            # write the uploaded PDFs to the temporary directory
            for file, save_path in zip(Files, save_paths):
                with open(save_path, mode='wb') as w:
                    w.write(file.getvalue())

            # once the save paths exist...
            if all(path.exists(save_path) for save_path in save_paths):
                # write a success message saying the files have been successfully saved
                st.success(f'Files {", ".join(file.name for file in Files)} are successfully saved!')
                # creates a timer to time the length of the summarization task and starts the timer
                start = time.time()
                # running the summarization task, and outputting the results to the front end
                for save_path in save_paths:
                    st.write(Chunk_and_Summarize([str(save_path)]))
                # ending the timer
                end = time.time()
                # using the timer, we calculate the minutes and seconds it took to perform the summarization task
                seconds = int(((end - start) % 60))
                minutes = int((end - start) // 60)
                # string to highlight the amount of time taken to complete the summarization task
                total_time = f"""Time taken to generate summaries:
                Minutes: {minutes} Seconds: {round(seconds, 2)}"""
                # sidebar is created to display the total time taken to complete the summarization task
                with st.sidebar:
                    st.header(total_time)

