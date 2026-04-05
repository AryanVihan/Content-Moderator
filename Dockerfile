FROM python:3.11-slim

# Install system dependencies (curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installation
RUN pip install uv

WORKDIR /app

# Copy dependency files first (layer caching)
COPY requirements.txt .

# Install dependencies with uv (significantly faster than pip)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY server/ ./server/
COPY openenv.yaml .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
EXPOSE 8080

# Health check for HuggingFace deployment
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["sh", "-c", \
     "if [ \"$MCP_MODE\" = 'true' ]; then \
        python server/mcp_server.py; \
      else \
        uvicorn server.main:app --host 0.0.0.0 --port 7860 \
        --workers 1 --timeout-keep-alive 30; \
      fi"]
