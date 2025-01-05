# NiNa

NiNa is a local environment setup for experimenting with Large Language Models (LLMs).
This Ui includes a Retrieval-Augmented Generation component which you can use to utilize your own local pdfs to your liking to provide context for your LLM.

For example i like to include Forensic Reports into my local RAG/PDF Folder and then ask the llm questions about it.

Feel free to use my finetuned llama 3.1 LLM NiNa: https://huggingface.co/LockeLamora2077/NiNa and let me know what you think :)


## Overview


Modular LLM Integration: Easily switch between different LLMs by updating environment variables.
RAG with Local PDFs: The included Docker services can handle retrieval-augmented generation by using local PDFs for context. Simply drop your PDFs into the specified folder and the retrieval service will index and serve their content.
Containerized Services: All components run within Docker containers, ensuring consistency and simplifying setup.
Local Integration with Ollama: Ollama runs on your host machine, providing easy model management, retrieval, and usage.

You can optionally enhance the llm response with data from abuse.ch by providing an ABUSE_API_KEY. 
If there is a file hash in the prompt to the LLM, the response will be enhanced with data from abuse.ch.
Get API Key here: https://auth.abuse.ch/user/me


## Prerequisites

1. **Ollama Installed Locally**:  
   Make sure you have Ollama installed on your host machine. Refer to the [Ollama installation guide](https://docs.ollama.ai/getting-started/installation) for details on your platform.

2. **LLM Downloaded via Ollama**:  
   Before running the Docker environment, you need to download (pull) the LLM model of your choice. For example, to pull hf.co/LockeLamora2077/NiNa model:
   ```bash
   ollama pull hf.co/LockeLamora2077/NiNaa
   ```
   
   Adjust this command as needed for your preferred model.

3. **Docker and Docker Compose**:  
   Ensure you have Docker and Docker Compose installed:
   - [Install Docker](https://docs.docker.com/get-docker/)
   - [Install Docker Compose](https://docs.docker.com/compose/install/)
  
## Easiest Way to start:
1. Make sure Ollama is installed and that you have pulled the LLM you want to use.
2. Clone this Repository with git clone
3. Edit the docker-compose.yml
4. Make sure that under environment variables the correct model name of your local model is listed:
   ```
    environment:
      - MODEL_NAME=hf.co/LockeLamora2077/NiNa  # Specify model
   ```
5. Change the Volume mapping within the Docker-Compose file to your local pdf folder:
  ```
    volumes:
      - 'your_pdf_folder_path':/app/pdf  # Mount PDFs directory from your local directory
```
6. Use docker-compose up to start the container
7. The NiNa UI with the RAG Component is now reachable at localhost:8000
8. If you just want to use my finetuned LLM without RAG, you can just download my LLM via Ollama pull and then use a UI like for Example OpenWebUI: https://github.com/open-webui/open-webui

## Configuration

1. **ENVIRONMENT Variables**  

   LLM_NAME=hf.co/LockeLamora2077/NiNaa
   LLM_HOST=host.docker.internal
   ABUSE_API_KEY='your_abuse_ch_key_here'
   ```

   - **LLM_NAME**: The name of the model you pulled with Ollama.
   - **LLM_HOST**: Should be set to `host.docker.internal` (for macOS/Windows) or `172.17.0.1` (on Linux) so the containers can communicate with your host machine’s Ollama instance.
   - **ABUSE_API_KEY**: Optional abuse.ch API Key, get it here: https://auth.abuse.ch/user/me

2. **Update Docker Compose**  
   Review `docker-compose.yml` and ensure it references `LLM_NAME`, `LLM_HOST`, and the volume mapping to make sure your local pdf folder is acessible. 

## Usage

1. **Start the Services**  
   Run the following command in the root directory:
   ```bash
   docker-compose up --build
   ```

   This command builds and starts the containers, integrating them with your locally running Ollama instance.

2. **Interacting with the LLM**  
    As soon as the Container is running the NiNa UI will be served at localhost:8000

3. **Stopping the Services**  
   To stop the containers, press `Ctrl+C` in the terminal where `docker-compose` is running. Alternatively, run:
   ```bash
   docker-compose down
   ```

## Contributing

Contributions are welcome! Feel free to open issues, suggest improvements, or submit pull requests.
I would also be interested in ideas for usecases in the #DFIR World!
