"""Esquemas (formas de los datos) de la API.

El formato de entrada de /predict se CONSTRUYE a partir de model/metadata.json:
  - tipo de cada variable (entero / decimal / categórica),
  - rango mínimo y máximo para las numéricas,
  - lista cerrada de valores para las categóricas.
Así, si se reentrena el modelo y cambia metadata.json, la validación se actualiza sola.
"""
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

RUTA_METADATA = Path(__file__).parent / "model" / "metadata.json"

if not RUTA_METADATA.exists():
    raise FileNotFoundError("Falta model/metadata.json (se genera con la última celda del notebook)")
_META = json.loads(RUTA_METADATA.read_text(encoding="utf-8"))

# Variables que el cliente puede omitir. Vacío = TODAS las variables de entrada son obligatorias.
# (metadata.json marca todas como "acepta_nulo" porque el Pipeline sabe imputar, pero la API exige
# el dato completo para no "adivinar" valores. Para permitir omitir alguna, añade su nombre aquí,
# p. ej. {"Age", "Embarked"}.)
OPCIONALES: set[str] = set()

# Reglas por variable (se usan para construir el esquema y para redactar los mensajes de error)
RANGOS: dict[str, tuple] = {}        # numéricas: nombre -> (mínimo, máximo)
CATEGORIAS: dict[str, list] = {}     # categóricas: nombre -> valores permitidos
TIPOS_ENTEROS: set[str] = set()      # numéricas que deben ser enteras


def _construir_campos() -> dict:
    """Convierte variables_entrada de metadata.json en campos de Pydantic."""
    campos = {}
    for var in _META["variables_entrada"]:
        nombre = var["nombre"]
        if var["tipo"] == "categorica":
            CATEGORIAS[nombre] = list(var["categorias_validas"])
            tipo = Literal[tuple(CATEGORIAS[nombre])]          # lista cerrada: cualquier otro valor se rechaza
            reglas = {"description": "Valores permitidos: " + ", ".join(CATEGORIAS[nombre])}
        else:
            es_entero = var["tipo"] == "entero"
            minimo = int(var["min"]) if es_entero else var["min"]
            maximo = int(var["max"]) if es_entero else var["max"]
            RANGOS[nombre] = (minimo, maximo)
            if es_entero:
                TIPOS_ENTEROS.add(nombre)
            tipo = int if es_entero else float
            # strict=True: no se aceptan textos ("3") ni booleanos (true) disfrazados de número
            reglas = {"ge": minimo, "le": maximo, "strict": True, "description": f"Rango válido: {minimo} a {maximo}"}
            if not es_entero:
                reglas["allow_inf_nan"] = False    # NaN e infinito no son valores válidos

        if nombre in OPCIONALES:
            campos[nombre] = (Optional[tipo], Field(None, **reglas))   # opcional: por defecto None
        else:
            campos[nombre] = (tipo, Field(..., **reglas))              # obligatorio
    return campos


# extra="forbid": cualquier campo que no esté definido en metadata.json se rechaza
PasajeroEntrada = create_model(
    "PasajeroEntrada",
    __config__=ConfigDict(extra="forbid"),
    __doc__="Variables de entrada del modelo (ya transformadas, tal como las espera el Pipeline).",
    **_construir_campos(),
)

# Ejemplo que aparece en /docs; se valida al importar para detectar a tiempo si deja de ser coherente con metadata.json
EJEMPLO = {"Pclass": 3, "Sex": 1, "Age": 30.0, "SibSp": 1, "Parch": 1, "Fare_log": 3.22, "HasCabin": 0, "Embarked": "S"}
PasajeroEntrada(**EJEMPLO)
PasajeroEntrada.model_config["json_schema_extra"] = {"example": EJEMPLO}


class PrediccionSalida(BaseModel):
    """Respuesta de /predict."""

    prediccion: int = Field(..., description="1 = sobrevive, 0 = no sobrevive")
    etiqueta: str = Field(..., description="Texto de la predicción")
    probabilidad_sobrevivir: float = Field(..., description="Probabilidad estimada de sobrevivir (0 a 1)")
    version_modelo: str = Field(..., description="Versión del modelo que respondió")


class EstadoSalida(BaseModel):
    """Respuesta de /health (a propósito, sin datos del modelo)."""

    status: str


def traducir_errores(errores: list[dict]) -> list[dict]:
    """Convierte los errores de validación de Pydantic en mensajes claros en español.

    Solo se devuelve el nombre del campo y un mensaje: NO se repite el valor enviado por el cliente
    ni se incluyen enlaces, rutas, tipos internos ni detalles de la librería.
    """
    salida = []
    for e in errores:
        tipo = e.get("type", "")
        # loc = ("body", "Age"): se descarta "body" y las posiciones numéricas del parser JSON
        partes = [str(p) for p in e.get("loc", ()) if p != "body" and not isinstance(p, int)]
        campo = ".".join(partes) or "cuerpo"

        if tipo == "missing":
            mensaje = "Campo obligatorio"
        elif tipo == "extra_forbidden":
            mensaje = "Campo no permitido"
        elif tipo in ("greater_than_equal", "less_than_equal") and campo in RANGOS:
            mensaje = f"Fuera de rango: debe estar entre {RANGOS[campo][0]} y {RANGOS[campo][1]}"
        elif tipo == "literal_error" and campo in CATEGORIAS:
            mensaje = "Valor no permitido. Opciones válidas: " + ", ".join(CATEGORIAS[campo])
        elif campo in TIPOS_ENTEROS:
            mensaje = "Debe ser un número entero (sin comillas ni decimales)"
        elif campo in RANGOS:
            mensaje = "Debe ser un número (sin comillas)"
        elif campo in CATEGORIAS:
            mensaje = "Valor no permitido. Opciones válidas: " + ", ".join(CATEGORIAS[campo])
        elif campo == "cuerpo":
            mensaje = "El cuerpo de la petición debe ser un objeto JSON válido"
        else:
            mensaje = "Valor no válido"
        salida.append({"campo": campo, "mensaje": mensaje})
    return salida
