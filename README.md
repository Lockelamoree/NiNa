# NiNa

NiNa is a local environment setup for experimenting with Large Language Models (LLMs) using [Ollama](https://docs.ollama.ai/) as a backend. This repository provides a Docker Compose configuration and a basic framework to integrate your chosen LLM model into a containerized environment.In addition, NiNa supports a basic Retrieval Augmented Generation (RAG) setup that utilizes a local PDF folder as the document source for context retrieval.

## Overview
This project aims to simplify local experimentation and development workflows with LLMs. By leveraging Docker Compose, it encapsulates dependencies, ensuring a smoother start-up process and easier model swapping. It is designed to work alongside Ollama, a tool that streamlines model management and provides a straightforward API for interacting with large language models locally.

Key features:

Modular LLM Integration: Easily switch between different LLMs by updating environment variables.
RAG with Local PDFs: The included Docker services can handle retrieval-augmented generation by using local PDFs for context. Simply drop your PDFs into the specified folder and the retrieval service will index and serve their content.
Containerized Services: All components run within Docker containers, ensuring consistency and simplifying setup.
Local Integration with Ollama: Ollama runs on your host machine, providing easy model management, retrieval, and usage.

## Prerequisites

1. **Ollama Installed Locally**:  
   Make sure you have Ollama installed on your host machine. Refer to the [Ollama installation guide](https://docs.ollama.ai/getting-started/installation) for details on your platform.

2. **LLM Downloaded via Ollama**:  
   Before running the Docker environment, you need to download (pull) the LLM model of your choice. For example, to pull a `llama2-7b` model:
   ```bash
   ollama pull hf.co/LockeLamora2077/NiNaa
   ```
   
   Adjust this command as needed for your preferred model.

3. **Docker and Docker Compose**:  
   Ensure you have Docker and Docker Compose installed:
   - [Install Docker](https://docs.docker.com/get-docker/)
   - [Install Docker Compose](https://docs.docker.com/compose/install/)

## Configuration

1. **.env File**  
   Create a `.env` file in the project root directory if it does not exist. This file holds environment variables used by Docker Compose.

   Example `.env`:
   ```env
   LLM_NAME=hf.co/LockeLamora2077/NiNaa
   LLM_HOST=host.docker.internal
  
   ```

   - **LLM_NAME**: The name of the model you pulled with Ollama.
   - **LLM_HOST**: Should be set to `host.docker.internal` (for macOS/Windows) or `172.17.0.1` (on Linux) so the containers can communicate with your host machine’s Ollama instance.

2. **Update Docker Compose**  
   Review `docker-compose.yml` and ensure it references `LLM_NAME`, `LLM_HOST`, and `LLM_PORT` from your `.env` file. This ensures that your chosen model will be integrated into the running environment.

## Usage

1. **Start the Services**  
   Run the following command in the root directory:
   ```bash
   docker-compose up --build
   ```

   This command builds and starts the containers, integrating them with your locally running Ollama instance.

2. **Interacting with the LLM**  
   Once the services are up, you can interact with the LLM through the exposed endpoints or via the integrated application logic. Refer to the code or documentation within this repository for specific usage examples and endpoints.

3. **Stopping the Services**  
   To stop the containers, press `Ctrl+C` in the terminal where `docker-compose` is running. Alternatively, run:
   ```bash
   docker-compose down
   ```

## Troubleshooting

- **Model not found**:  
  Ensure the `LLM_NAME` matches a model you have pulled with Ollama.
  
- **Connection issues**:  
  Verify that `LLM_HOST` is set correctly and that Ollama is running. Try `ollama list` to ensure your model is available.

- **Port conflicts**:  
  Change `LLM_PORT` in `.env` if another service is using the same port.

## Contributing

Contributions are welcome! Feel free to open issues, suggest improvements, or submit pull requests.
