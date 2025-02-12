# syntax=docker/dockerfile:1.2
FROM python:3.12-slim
# put you docker configuration here
# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Actualizar paquetes del sistema e instalar dependencias esenciales
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar los archivos de dependencias antes del código para aprovechar la caché de Docker
COPY requirements.txt requirements-dev.txt requirements-test.txt ./

# Instalar dependencias de producción
RUN pip install --no-cache-dir -r requirements.txt

# Instalar dependencias de desarrollo y testeo si es necesario
ARG INSTALL_DEV=false
RUN if [ "$INSTALL_DEV" = "true" ]; then pip install --no-cache-dir -r requirements-dev.txt -r requirements-test.txt; fi

# Copiar todo el código del proyecto
COPY . .

# Exponer el puerto en el que correrá la API (por defecto FastAPI usa 8000)
EXPOSE 8080

# Comando por defecto para iniciar la API
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]