# Multi-stage: Build deps, then runtime (use Python 3.9)
FROM python:3.9-slim AS builder

WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.9-slim

# Install runtime deps minimally
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy package and metadata
COPY src/ src/
COPY pyproject.toml README.md ./

# Expose for CLI
ENTRYPOINT ["python", "-m", "llmsays"]
CMD ["--help"]

# Usage:
# docker build -t llmsays . 
# docker run -e OPENROUTER_API_KEY=sk-... llmsays "What is 2+2?"