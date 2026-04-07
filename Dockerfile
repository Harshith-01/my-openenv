FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment setup
ENV HOST=0.0.0.0
ENV PORT=7860

EXPOSE 7860

# Run the FastAPI server which acts as the environment
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
