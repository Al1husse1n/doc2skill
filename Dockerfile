FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY doc2skill ./doc2skill

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "doc2skill.web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]

