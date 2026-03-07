FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit matplotlib seaborn plotly

COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY config.py .

EXPOSE 8501

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
