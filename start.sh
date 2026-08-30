#!/bin/bash
# Script para arrancar la aplicación SAM 3 Histology (Frontend + Backend)

# Colores para la salida
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0;37m' # No Color

echo -e "${CYAN}===================================================${NC}"
echo -e "${CYAN}      SAM 3 Histology Segmenter - Launching...     ${NC}"
echo -e "${CYAN}===================================================${NC}"

# Limpiar puertos si ya están en uso
fuser -k 8000/tcp 4321/tcp 2>/dev/null || true
sleep 1

# 1. Iniciar el Backend FastAPI
echo -e "${YELLOW}[1/2] Iniciando Backend FastAPI en el puerto 8000...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: El entorno virtual .venv no existe. Por favor, corre la instalación primero.${NC}"
    exit 1
fi

PYTHONPATH=backend:sam3 ./.venv/bin/python backend/main.py &
BACKEND_PID=$!

# Esperar a que el backend inicialice el modelo SAM 3 y responda HTTP
echo -e "${YELLOW}Esperando inicialización del backend y carga del modelo SAM 3 en GPU...${NC}"
READY=false
for i in {1..30}; do
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo -e "${RED}Error: El backend falló al iniciar. Revisa los logs.${NC}"
        exit 1
    fi
    if curl -s http://127.0.0.1:8000/api/ontologies >/dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 1
done

if [ "$READY" = true ]; then
    echo -e "${GREEN}✔ Backend y modelo SAM 3 listos (PID: $BACKEND_PID)${NC}"
else
    echo -e "${YELLOW}⚠ El backend sigue iniciando, continuando con el frontend...${NC}"
fi

# 2. Iniciar el Frontend Astro
echo -e "${YELLOW}[2/2] Iniciando Frontend Astro en el puerto 4321...${NC}"
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}Instalando dependencias de node...${NC}"
    cd frontend && npm install && cd ..
fi

cd frontend
npm run dev -- --port 4321 --force &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}✔ Frontend corriendo con PID: $FRONTEND_PID${NC}"
echo -e "${GREEN}===================================================${NC}"
echo -e "${GREEN}Servidores corriendo con éxito:${NC}"
echo -e "  - ${CYAN}Frontend:${NC} http://localhost:4321"
echo -e "  - ${CYAN}Backend API:${NC} http://localhost:8000"
echo -e "Presiona ${YELLOW}CTRL+C${NC} para detener ambos servidores."
echo -e "${GREEN}===================================================${NC}"

# Función de limpieza al salir con Ctrl+C
cleanup() {
    echo -e "\n${YELLOW}Deteniendo servidores...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}✔ Servidores detenidos. ¡Hasta luego!${NC}"
    exit 0
}

trap cleanup SIGINT

# Esperar a que terminen los procesos en segundo plano
wait
