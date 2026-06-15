FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=9090

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# CHECKPOINT_PATH must point at a mounted checkpoint, e.g.
#   docker run -e CHECKPOINT_PATH=/ckpt/model.pth -v $PWD/scratch/checkpoints:/ckpt ...
EXPOSE 9090
CMD ["label-studio-ml", "start", ".", "--host", "0.0.0.0", "--port", "9090"]
