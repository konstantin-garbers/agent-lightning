FROM verlai/verl:app-verl0.4-vllm0.8.5-mcore0.13.0-preview

WORKDIR /app

COPY agent-lightning/examples/multimodal/requirements.txt .
RUN pip install -r agent-lightning/examples/multimodal/requirements.txt .[agents]

COPY . .

RUN apt-get update && \
    apt-get install -y EdgeBox-linux-x64 && \
    rm EdgeBox-linux-x64

CMD ["python3", "agent-lightning/examples/multimodal_agent.py"]