import openai
import streamlit as st
from openai import AzureOpenAI
import os
import numpy as np
import json
from datetime import datetime
import pandas as pd
from azure.cosmos import CosmosClient, exceptions, PartitionKey
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Ignite 2024 Demo", layout="wide", initial_sidebar_state="expanded")

# UI text strings
page_title = "Cosmos DB DiskANN vs QFLAT Index"
page_helper = "The Streamlit app uses cosine similarity to semantically match your query with records and find matching properties"
empty_search_helper = "Select a neighborhood in Seattle, and enter a search term to get started."
semantic_search_header = "What are you looking for?"
semantic_search_placeholder = "House with a beach view, pet-friendly, near downtown"
search_label = "Search for listings"
venue_list_header = "Listing details"

# Initialize global variables for Cosmos DB client, database, and containers
if "cosmos_client" not in st.session_state:
    endpoint = os.getenv("AZURE_COSMOSDB_ENDPOINT")
    key = os.getenv("AZURE_COSMOSDB_KEY")
    st.session_state.cosmos_client = CosmosClient(endpoint, credential=key)
    database_name = 'diskanndb'  # Replace with your database name
    st.session_state.cosmos_database = st.session_state.cosmos_client.create_database_if_not_exists(database_name)

    # Define the vector property and dimensions
    cosmos_vector_property = "embedding"
    openai_embeddings_dimensions = 1536

    # Create listings_search container without any index
    container_name = 'search'
    st.session_state.cosmos_container = st.session_state.cosmos_database.create_container_if_not_exists(
        id=container_name,
        partition_key=PartitionKey(path="/id"),
        vector_embedding_policy={
            "vectorEmbeddings": [
                {
                    "path": "/" + cosmos_vector_property,
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": openai_embeddings_dimensions
                },
            ]
        },
        offer_throughput=50000
    )

    # Create listings_search_qflat container with QFLAT vector index
    container_name_qflat = 'search_qflat'
    st.session_state.cosmos_container_qflat = st.session_state.cosmos_database.create_container_if_not_exists(
        id=container_name_qflat,
        partition_key=PartitionKey(path="/id"),
        indexing_policy={
            "includedPaths": [
                {
                    "path": "/*"
                }
            ],
            "excludedPaths": [
                {
                    "path": "/\"_etag\"/?",
                    "path": "/" + cosmos_vector_property + "/*",
                }
            ],
            "vectorIndexes": [
                {
                    "path": "/" + cosmos_vector_property,
                    "type": "quantizedFlat",
                }
            ]
        },
        vector_embedding_policy={
            "vectorEmbeddings": [
                {
                    "path": "/" + cosmos_vector_property,
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": openai_embeddings_dimensions
                },
            ]
        },
        offer_throughput=50000
    )

    # Create listings_search_diskann container with DiskANN vector index
    container_name_diskann = 'search_diskann'
    st.session_state.cosmos_container_diskann = st.session_state.cosmos_database.create_container_if_not_exists(
        id=container_name_diskann,
        partition_key=PartitionKey(path="/id"),
        indexing_policy={
            "includedPaths": [
                {
                    "path": "/*"
                }
            ],
            "excludedPaths": [
                {
                    "path": "/\"_etag\"/?",
                    "path": "/" + cosmos_vector_property + "/*",
                }
            ],
            "vectorIndexes": [
                {
                    "path": "/" + cosmos_vector_property,
                    "type": "diskANN",
                }
            ]
        },
        vector_embedding_policy={
            "vectorEmbeddings": [
                {
                    "path": "/" + cosmos_vector_property,
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": openai_embeddings_dimensions
                },
            ]
        },
        offer_throughput=50000
    )

# Function to log times
def log_time(start):
    end = time.perf_counter()
    elapsed_time = end - start
    return f"{elapsed_time:.4f} seconds"

# Initialize the embedding client only once
if "embedding_client" not in st.session_state:
    st.session_state.embedding_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_APIKEY"),
        api_version="2023-05-15",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

# Handler functions
def embedding_query(text_input):
    print("text_input", text_input)
    start_time = time.perf_counter()
    response = st.session_state.embedding_client.embeddings.create(
        input=text_input,
        model="text-embedding-ada-002"  # Use the appropriate model
    )

    json_response = response.model_dump_json(indent=2)
    parsed_response = json.loads(json_response)
    embedding = parsed_response['data'][0]['embedding']
    st.session_state.embedding_gen_time = log_time(start_time)
    print(f"Embedding generation time: {st.session_state.embedding_gen_time}")
    return embedding

def handler_search_listings(indices, ask):
    emb = embedding_query(ask)
    num_results = 10

    # Query strings
    listings_search_query = '''
    SELECT TOP @num_results l.id, l.title, l.abstract, VectorDistance(l.embedding, @emb) as SimilarityScore
    FROM l
    ORDER BY VectorDistance(l.embedding,@emb)
    '''

    container = {
        'No Index': st.session_state.cosmos_container,
        'QFLAT Index': st.session_state.cosmos_container_qflat,
        'DiskANN Index': st.session_state.cosmos_container_diskann
    }.get(indices)

    parameters = [
        {"name": "@num_results", "value": num_results},
        {"name": "@emb", "value": emb}
    ]

    try:
        start_time = time.perf_counter()  # Capture start time
        results = container.query_items(listings_search_query, parameters=parameters, enable_cross_partition_query=True)
        results_list = list(results)
        elapsed_time = log_time(start_time)
        st.session_state.suggested_listings = pd.DataFrame(results_list)
        st.session_state.query_time = elapsed_time
        st.session_state.ru_consumed = container.client_connection.last_response_headers['x-ms-request-charge']
    except exceptions.CosmosHttpResponseError as e:
        st.error(f"An error occurred: {e}")

# UI elements
def render_cta_link(url, label, font_awesome_icon):
    st.markdown(
        '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">',
        unsafe_allow_html=True)
    button_code = f'''<a href="{url}" target=_blank><i class="fa {font_awesome_icon}"></i> {label}</a>'''
    return st.markdown(button_code, unsafe_allow_html=True)

def render_search():
    search_disabled = True
    with st.sidebar:
        st.selectbox(label="Index", options=['No Index', 'QFLAT Index', 'DiskANN Index'], index=0, key="index_selection")
        st.text_input(label=semantic_search_header, placeholder=semantic_search_placeholder, key="user_category_query")

        if "user_category_query" in st.session_state and st.session_state.user_category_query != "":
            search_disabled = False

        st.button(label=search_label, key="location_search", disabled=search_disabled,
                  on_click=handler_search_listings, args=(st.session_state.index_selection, st.session_state.user_category_query))

        st.write("---")
        render_cta_link(url="https://twitter.com/abeomor", label="X", font_awesome_icon="fa-twitter")
        render_cta_link(url="https://linkedin.com/in/abeomor", label="LinkedIn", font_awesome_icon="fa-linkedin")
        render_cta_link(url="https://github.com/AbeOmor", label="GitHub", font_awesome_icon="fa-github")

def render_search_result():
    col1 = st.container()
    col1.write(f"Found {len(st.session_state.suggested_listings)} listings.")
    col1.write(f"Embedding generation time: {st.session_state.embedding_gen_time}")
    col1.write(f"Query time: {st.session_state.query_time}")
    col1.write(f"RU consumed: {st.session_state.ru_consumed}")
    col1.table(st.session_state.suggested_listings)

# Main execution
render_search()

st.title(page_title)
st.write(page_helper)
st.write("---")

if "suggested_listings" not in st.session_state:
    st.write(empty_search_helper)
else:
    render_search_result()
