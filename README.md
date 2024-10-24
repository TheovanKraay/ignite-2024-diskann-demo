# Seattle Airbnb Rentals Listing

This repository contains a Streamlit application that allows users to search for Airbnb rentals in Seattle using semantic search powered by OpenAI embeddings and Azure Cosmos DB. The application uses cosine similarity to match user queries with Airbnb listings and displays the results.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Semantic search for Airbnb listings using OpenAI embeddings.
- Integration with Azure Cosmos DB for storing and querying listings.
- Support for different vector indexing methods (No Index, HNSW Index, DiskANN Index).
- Interactive UI built with Streamlit.

## Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/TheovanKraay/ai-search-demo.git
   ```
   
[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https://raw.githubusercontent.com/TheovanKraay/ignite-2024-diskann-demo/main/azuredeploy.json)
   