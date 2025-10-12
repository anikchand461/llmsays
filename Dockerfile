# Multi-stage build for llmsays with Python 3.9
FROM python:3.9-slim AS builder
WORKDIR /app

# Copy build files
COPY pyproject.toml requirements.txt ./
# Install build dependencies and package in builder (for wheels if needed)
RUN pip install --no-cache-dir --user -r requirements.txt build
# Optional: Build package artifacts (wheel/sdist) - uncomment if needed for CI artifacting
# COPY src/ src/
# RUN python -m build --sdist --wheel --outdir dist/

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy installed deps from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source and config
COPY src/ src/
COPY pyproject.toml README.md ./
COPY requirements.txt ./

# Install runtime dependencies and the package itself
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir . && \
    rm -rf /var/lib/apt/lists/*  # Clean up for smaller image

# Default command: Show CLI help
ENTRYPOINT ["llmsays"]
CMD ["--help"]

# Build: docker build -t llmsays .
# Run: docker run -e OPENROUTER_API_KEY=sk-... llmsays "What is 2+2?"
# Notes:
# - Image size: ~250MB (slim base + deps).
# - For dev: Add volume mount for src/ in docker run.
# - Security: Use non-root user if needed: Add `USER 1001` after install.