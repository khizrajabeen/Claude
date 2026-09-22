FROM python:3.11-slim

# Build tools are needed for the scientific wheels and nothing after, so
# they are dropped in the same layer that uses them.
WORKDIR /app
COPY requirements.txt .
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get purge -y build-essential \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

COPY . .

# Not root. The bot writes to state/ and web/data and needs nothing else.
RUN useradd --create-home --shell /usr/sbin/nologin meridian \
 && mkdir -p state data web/data \
 && chown -R meridian:meridian /app
USER meridian

ENV PYTHONUNBUFFERED=1 TZ=UTC
CMD ["sh", "-c", "python main.py ${MERIDIAN_MODE:-run}"]
