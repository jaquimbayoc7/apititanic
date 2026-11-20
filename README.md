# API de supervivencia del Titanic

API con FastAPI que sirve un modelo XGBoost (Pipeline de scikit-learn con imputación, escalado y codificación incluidos).
El modelo se carga **una sola vez** al arrancar el servidor.

> **Sobre el desempeño:** F1 de la clase "sobrevive" en el conjunto de prueba = **0.71** (ROC-AUC 0.80, 134 pasajeros).
> La muestra es pequeña (margen de error ≈ ±0.06). Es un ejercicio académico: no debe usarse para decisiones reales.

## Estructura

| Archivo | Para qué sirve |
|---|---|
| `main.py` | La app: carga del modelo y endpoints `/health` y `/predict` |
| `schemas.py` | Forma y límites de los datos de entrada y salida |
| `requirements.txt` | Dependencias con versiones exactas |
| `model/model.joblib` | Modelo entrenado (**se versiona en git**, Render lo necesita) |
| `model/metadata.json` | Ficha del modelo: versión, métrica, variables, rangos y versiones de librerías |

## Ejecutar en local

Requiere **Python 3.14.8** (la versión con la que se guardó el modelo; con otra versión de Python o de scikit-learn el `.joblib` puede fallar o dar resultados distintos).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# La llave vive en el archivo .env (API_KEY=...). Si no lo tienes, créalo desde la plantilla y cambia el valor:
copy .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"     # genera una llave para pegar en .env

uvicorn main:app --env-file .env --reload
```

**Sin `API_KEY` definida la API no arranca** (falla a propósito con un mensaje claro). La llave nunca se escribe en el código: `.env` está en `.gitignore` y no se sube a git. En el repositorio solo va `.env.example`, con un valor de mentira.

Alternativa sin archivo: `$env:API_KEY = "..."` en la terminal y luego `uvicorn main:app --reload` (Linux/Mac: `export API_KEY="..."`).

Abre <http://127.0.0.1:8000/docs> para probar los endpoints desde el navegador (botón **Authorize** → pega la llave).

## Autenticación
- `GET /health` es **público**.
- `POST /predict` exige el encabezado `X-API-Key` con la llave configurada en `API_KEY`.
- Si la llave falta o es incorrecta responde `401 {"detail": "No autorizado"}`, siempre el mismo mensaje.

## Endpoints

### `GET /health`
Comprueba que el servicio está vivo. No devuelve datos del modelo.
```json
{"status": "ok"}
```

### `POST /predict`
Recibe las variables **ya transformadas** (tal como las espera el modelo):

| Campo | Tipo | Valores válidos |
|---|---|---|
| `Pclass` | entero | 1, 2 o 3 |
| `Sex` | entero | 1 = mujer, 0 = hombre |
| `Age` | decimal | 0.42 a 80 |
| `SibSp` | entero | 0 a 8 |
| `Parch` | entero | 0 a 6 |
| `Fare_log` | decimal | 0 a 6.24. Es `log(1 + tarifa)`: se envía calculado, no la tarifa |
| `HasCabin` | entero | 1 = tiene cabina, 0 = no |
| `Embarked` | texto | `C`, `Q` o `S` |

**Todos los campos son obligatorios** (un `null` o un campo ausente devuelve 422). Para permitir omitir alguno, añade su nombre a `OPCIONALES` en `schemas.py`.

Ejemplo (PowerShell):
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" `
  -Headers @{"X-API-Key" = "TU_LLAVE_DEL_ARCHIVO_.env"} `
  -Body '{"Pclass":3,"Sex":1,"Age":30,"SibSp":1,"Parch":1,"Fare_log":3.22,"HasCabin":0,"Embarked":"S"}'
```
Ejemplo (curl):
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -H "X-API-Key: TU_LLAVE_DEL_ARCHIVO_.env" \
  -d '{"Pclass":3,"Sex":1,"Age":30,"SibSp":1,"Parch":1,"Fare_log":3.22,"HasCabin":0,"Embarked":"S"}'
```
Respuesta:
```json
{"prediccion": 1, "etiqueta": "sobrevive", "probabilidad_sobrevivir": 0.5604, "version_modelo": "1.0.0"}
```
- `prediccion` es 1 si la probabilidad es ≥ 0.5.
- Un valor fuera de rango, de tipo incorrecto, un campo desconocido o un campo obligatorio faltante devuelven **422**. La validación se construye desde `model/metadata.json`, y los errores no repiten el valor enviado ni exponen detalles internos:
```json
{"detail": "Datos de entrada no válidos",
 "errores": [{"campo": "Age", "mensaje": "Fuera de rango: debe estar entre 0.42 y 80.0"}]}
```

## Desplegar en Render

1. Sube la carpeta a un repositorio de GitHub **incluyendo `model/model.joblib` y `model/metadata.json`**.
2. En Render: **New → Web Service** y conecta el repositorio.
3. Configura:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path:** `/health`
   - **Variable de entorno `API_KEY`:** créala en **Environment** del servicio en Render con una llave generada (no la pongas en el repositorio). Sin ella el servicio no inicia.
   - **Variable de entorno:** `PYTHON_VERSION` = `3.14.8` (debe coincidir con la del modelo; si Render no ofrece esa versión, vuelve a generar el modelo con una versión que sí soporte y actualiza `requirements.txt`).
4. Si el repositorio contiene más cosas que la API (notebook, `Data/`), no estorban, pero el servicio solo necesita `main.py`, `schemas.py`, `requirements.txt` y `model/`.

## Actualizar el modelo
Vuelve a ejecutar la última celda del notebook (genera `model/model.joblib` y `model/metadata.json`), copia las versiones nuevas de `metadata.json` a `requirements.txt` y haz commit de los dos archivos de `model/`.
