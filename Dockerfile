FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies if requirements.txt is non-empty
RUN pip install --upgrade pip setuptools wheel \
 && if [ -s /app/requirements.txt ]; then pip install -r /app/requirements.txt; fi

# Copy project sources
COPY . /app

# Create a non-root user and give ownership of app directory
RUN groupadd -r app && useradd -r -g app app \
 && chown -R app:app /app

USER app

# Expose a typical ML/web port; change if your app uses another port
EXPOSE 5000

# Default command: drop to a shell. Replace with your project's run command.
CMD ["python", "main.py"]