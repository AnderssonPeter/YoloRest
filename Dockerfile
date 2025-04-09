ARG PYTHON_VERSION="3.11.9"
ARG CPU_ARCHITECTURE="amd64"
ARG DEBIAN_VERSION="bookworm"
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION}
ARG CPU_ARCHITECTURE
ARG DEBIAN_VERSION
ARG LIBEDGETPU_VERSION1="16.0"
ARG LIBEDGETPU_VERSION2="2.17.1-1"
ARG LIBEDGETPU_RELEASE_URL="https://github.com/feranick/libedgetpu/releases/download/${LIBEDGETPU_VERSION1}TF${LIBEDGETPU_VERSION2}/libedgetpu1-max_${LIBEDGETPU_VERSION1}tf${LIBEDGETPU_VERSION2}.${DEBIAN_VERSION}_${CPU_ARCHITECTURE}.deb"

# Install libedgetpu
RUN apt-get update && apt-get install libusb-1.0-0 curl -y \
    && curl ${LIBEDGETPU_RELEASE_URL} --output /tmp/libedgetpu.deb -L --fail \
    && echo "yes" | dpkg -i /tmp/libedgetpu.deb \
    && rm /tmp/libedgetpu.deb && rm -rf /var/lib/apt/lists/*

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copy the source code into the container.
COPY . .

RUN pip install -r requirements.txt --no-cache-dir

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD [ "curl", "--fail", "--silent", "http://localhost:8000/health" ]

# Run the application.
ENTRYPOINT ["python", "main.py"]