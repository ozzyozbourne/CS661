FROM continuumio/miniconda3

WORKDIR /app

COPY env.yml .
RUN conda env create -f env.yml

# Activate conda environment
SHELL ["conda", "run", "-n", "finals", "/bin/bash", "-c"]

# Copy application files
COPY app.py .
COPY diabetes.csv .

# Expose port for Streamlit
EXPOSE 8501

# Run Streamlit
ENTRYPOINT ["conda", "run", "-n", "finals", "streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
