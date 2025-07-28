# Dockerfile

FROM adobe-base

WORKDIR /app

COPY . .

ENTRYPOINT ["python", "main.py"]
