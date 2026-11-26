FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies for mDNS (Avahi), audio (PortAudio/ALSA), and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    avahi-utils \
    libportaudio2 \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir -e ".[antigravity,hardware-mic]" || pip install --no-cache-dir -e ".[hardware-mic]"

EXPOSE 8765

CMD ["agy-home-bridge", "serve", "--host", "0.0.0.0", "--port", "8765"]
