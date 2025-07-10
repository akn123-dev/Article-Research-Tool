## Article Research Tool

### Overview

The Article Research Tool is a Streamlit-based application that enables users to extract, store, and retrieve information from online articles. It leverages LangChain, OpenAI embeddings, and FAISS for efficient data retrieval and question answering.

### Features

Load and process URLs: Users can input up to three URLs to extract text content.

Vector-based search: The extracted text is embedded using OpenAI embeddings and stored in a FAISS vector database.

Natural language query support: Users can enter queries, and the system retrieves relevant answers along with source references.

### Technology Stack

Python: Programming language used.

Streamlit: Frontend UI for user interaction.

LangChain: Framework for handling language model interactions.

OpenAI GPT: Language model for processing queries.

FAISS (Facebook AI Similarity Search): Vector storage and retrieval system.

### Installation

Prerequisites

Ensure you have Python installed. Recommended version: Python 3.8+.

### Steps

-Clone the repository
git clone <repository_url>
cd <repository_folder>


-Set up the OpenAI API key
export OPENAI_API_KEY='your_openai_api_key_here'


### Usage

- Run the Streamlit app
streamlit run main.py

Enter up to three article URLs in the sidebar.

Click Run the model to process the articles.

Input a query in the QUESTION field to retrieve information from the processed documents.


### Potential Enhancements

Expand support for more document formats (PDF, TXT, etc.).

Improve error handling and logging.

Implement caching for faster query responses.

Allow batch processing of multiple documents.



Contributors

Akhil Nair - Developer

