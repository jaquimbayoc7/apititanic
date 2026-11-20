"""API FastAPI que sirve el modelo guardado en model/model.joblib.

Ejecutar en local:  uvicorn main:app --reload
Documentación interactiva: http://127.0.0.1:8000/docs
"""
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from schemas import EstadoSalida, PasajeroEntrada, PrediccionSalida, traducir_errores

logger = logging.getLogger("api")

# Rutas relativas a este archivo (así funcionan igual en local y en Render)
CARPETA_MODELO = Path(__file__).parent / "model"
UMBRAL = 0.5  # probabilidad mínima para predecir "sobrevive" (el mismo umbral del notebook)


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Se ejecuta UNA sola vez al arrancar: lee la API key y carga el modelo y su metadata en memoria."""
    # La llave correcta viene SOLO de la variable de entorno API_KEY (nunca escrita en el código).
    # Si no está definida (o está vacía), la app NO arranca: así nunca queda /predict sin protección.
    api_key = os.environ.get("API_KEY", "")
    if not api_key.strip():
        raise RuntimeError("La variable de entorno API_KEY no está definida: la API no puede iniciar sin ella")
    app.state.api_key = api_key  # solo en memoria; no se escribe en logs ni se devuelve en ninguna respuesta

    # Si falta algún archivo, esto lanza un error y la app no arranca (falla pronto y con mensaje claro)
    app.state.modelo = joblib.load(CARPETA_MODELO / "model.joblib")
    app.state.metadata = json.loads((CARPETA_MODELO / "metadata.json").read_text(encoding="utf-8"))
    logger.info("Modelo cargado: %s v%s", app.state.metadata["modelo"]["nombre"],
                app.state.metadata["modelo"]["version"])
    yield
    # (no hay nada que liberar al apagar)


DESCRIPCION = """
API que predice si un pasajero del Titanic habría sobrevivido, usando un modelo XGBoost.

**Autor:** jaquimbayoc7
**Institución:** Universidad Central

## Diccionario de datos

### Entrada (`POST /predict`)

| Variable | Tipo | Rango / valores | Descripción |
|---|---|---|---|
| `Pclass` | entero | 1 a 3 | Clase del pasaje (1 = primera, 3 = tercera) |
| `Sex` | entero | 0 o 1 | Sexo del pasajero, codificado en 0/1 |
| `Age` | decimal | 0.42 a 80 | Edad en años |
| `SibSp` | entero | 0 a 8 | Número de hermanos/cónyuges a bordo |
| `Parch` | entero | 0 a 6 | Número de padres/hijos a bordo |
| `Fare_log` | decimal | 0 a 6.24 | Logaritmo de la tarifa pagada |
| `HasCabin` | entero | 0 o 1 | 1 si tiene cabina registrada, 0 si no |
| `Embarked` | texto | ver `/docs` | Puerto de embarque |

### Salida

| Campo | Tipo | Descripción |
|---|---|---|
| `prediccion` | entero | 1 = sobrevive, 0 = no sobrevive |
| `etiqueta` | texto | Texto de la predicción |
| `probabilidad_sobrevivir` | decimal | Probabilidad estimada de sobrevivir (0 a 1) |
| `version_modelo` | texto | Versión del modelo que respondió |
"""

app = FastAPI(
    title="API de supervivencia del Titanic",
    version="1.0.0",
    description=DESCRIPCION,
    lifespan=ciclo_de_vida,
)

# La llave se envía en el encabezado X-API-Key. auto_error=False: el rechazo lo decidimos nosotros
# (siempre el mismo 401 genérico) y, además, /docs muestra el botón "Authorize" para probar con llave.
esquema_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


def exigir_api_key(request: Request, llave: Optional[str] = Security(esquema_api_key)) -> None:
    """Deja pasar solo si el encabezado X-API-Key coincide con la llave configurada."""
    # Se compara en bytes: compare_digest con texto no ASCII lanzaría un error. La comparación es
    # de tiempo constante (secrets.compare_digest), para que no se pueda adivinar la llave midiendo tiempos.
    es_valida = llave is not None and secrets.compare_digest(llave.encode("utf-8"),
                                                             request.app.state.api_key.encode("utf-8"))
    if not es_valida:
        # Mismo mensaje si la llave falta o es incorrecta: no se da pistas sobre cuál fue el caso
        raise HTTPException(status_code=401, detail="No autorizado", headers={"WWW-Authenticate": "ApiKey"})


@app.exception_handler(RequestValidationError)
async def entrada_invalida(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Errores de validación (422): mensajes claros en español, sin repetir lo que envió el cliente
    ni exponer detalles internos de la librería (enlaces, tipos, rutas)."""
    return JSONResponse(status_code=422,
                        content={"detail": "Datos de entrada no válidos", "errores": traducir_errores(exc.errors())})


@app.get("/health", response_model=EstadoSalida)
def health() -> dict:
    """Comprobación de que el servicio está vivo. No expone datos del modelo."""
    return {"status": "ok"}


@app.post("/predict", response_model=PrediccionSalida, dependencies=[Depends(exigir_api_key)])
def predict(pasajero: PasajeroEntrada, request: Request) -> dict:
    """Recibe las variables de un pasajero y devuelve la predicción, su probabilidad y la versión del modelo."""
    modelo = request.app.state.modelo
    meta = request.app.state.metadata

    # El Pipeline busca las columnas por nombre: se arma el DataFrame en el orden guardado en metadata.json.
    # Los opcionales ausentes (None) se pasan como NaN para que el imputador del Pipeline los rellene.
    fila = {k: (np.nan if v is None else v) for k, v in pasajero.model_dump().items()}
    X = pd.DataFrame([fila], columns=meta["orden_columnas"])

    try:
        prob = float(modelo.predict_proba(X)[0, 1])
    except Exception:  # no se filtra el detalle interno al cliente; queda en el log del servidor
        logger.exception("Fallo al predecir")
        raise HTTPException(status_code=500, detail="Error interno al calcular la predicción")

    clase = int(prob >= UMBRAL)
    return {
        "prediccion": clase,
        "etiqueta": "sobrevive" if clase else "no sobrevive",
        "probabilidad_sobrevivir": round(prob, 4),
        "version_modelo": meta["modelo"]["version"],
    }
