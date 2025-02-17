# syntax=docker/dockerfile:1.4
# Etapa de construcción (build stage)
FROM python:3.12-slim as builder

# Establecer el directorio de trabajo
WORKDIR /app

# Actualizar paquetes del sistema e instalar dependencias esenciales
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar los archivos de dependencias
COPY requirements.txt requirements-dev.txt requirements-test.txt ./

# Crear un entorno virtual e instalar dependencias de producción
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Instalar dependencias de desarrollo y testeo si es necesario
ARG INSTALL_DEV=false
RUN if [ "$INSTALL_DEV" = "true" ]; then \
    /opt/venv/bin/pip install --no-cache-dir -r requirements-dev.txt -r requirements-test.txt; \
    fi

# Copiar todo el código del proyecto
COPY . .

# Etapa final (production stage)
FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el entorno virtual desde la etapa de construcción
COPY --from=builder /opt/venv /opt/venv

# Crear un usuario no root para ejecutar la aplicación
RUN useradd -m myuser && chown -R myuser:myuser /app
USER myuser

# Copiar el código del proyecto
COPY --chown=myuser:myuser . .

# Exponer el puerto en el que correrá la API
EXPOSE 8080

# Usar el entorno virtual para ejecutar la aplicación
CMD ["/opt/venv/bin/uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]