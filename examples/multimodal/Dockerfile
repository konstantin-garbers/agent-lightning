# 1. Base Image
FROM verlai/verl:app-verl0.4-vllm0.8.5-mcore0.13.0-preview

WORKDIR /app

COPY . .

RUN pip install -r examples/multimodal/requirements.txt .[agents]

RUN apt-get update && \
    apt-get install -y EdgeBox-linux-x64 && \
    rm EdgeBox-linux-x64

COPY entrypoint.sh .
RUN chmod +x ./entrypoint.sh

CMD ["./entrypoint.sh"]