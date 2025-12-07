```Universidad San Francisco de Quito
Data Mining
Proyecto Activo Financiero
John Ochoa (00345743)

#Github Link:
https://github.com/johnnyredwood/Proyecto-ActivoFinanciero

#Descripción del proyecto:
Infraestructura analítica y de machine learning para un modelo de clasificación orientado a decisiones de trading intradía/simple sobre activos financieros. El proyecto integra:
- Jupyter para consumo de datos raw
- Script para construcción automatizada de tabla de datos en esquema analytics
- Notebook con EDA, preparación y entrenamiento del modelo.
- Simulación de inversión (backtest) 2025 con reglas explícitas y métricas de desempeño.
- FastAPI para exponer el modelo vía HTTP (`/health`, `/predict`) con Docker/Compose, montando el artefacto entrenado desde `libros/modelos`.

Los datos de activos financieros serán directamente extraídos de la API de Yahoo Finance para un rango de fechas especificado en las variables de entorno y a su vez identificado por los denominadores tickers de igual forma previamente indicados en el archivo .env (ver .env.example o descripción de variables de entorno del presente readme)

El modelo predice la dirección próxima del retorno (arriba/abajo) que es una variable target de tipo binario denominada target_up usando features derivadas (día de la semana, volatilidad reciente, retornos previos, volumen, etc.). La API devuelve únicamente la etiqueta `pred_label` para facilitar su consumo y de igual forma simplificar el uso del modelo.

#Checklist de aceptación
[x] Docker Compose levanta la API del modelo (FastAPI) correctamente.
[x] Variables sensibles y puertos gestionados vía `.env` y `docker-compose.yaml`.
[x] Artefacto del modelo guardado en `libros/modelos` y montado en el contenedor.
[x] Notebook con EDA breve (balance de clases, distribución de retornos).
[x] Balanceo de clases en Train (undersampling/oversampling configurable).
[x] Obtención de mejor modelo a través de validación y testing en 7 modelos de Machine Learning
[x] Simulación 2025 con curva de equity, drawdown y comparación con métricas ML.
[x] README claro: pasos, variables, arquitectura, decisiones y troubleshooting.

#Variables de ambiente: listado y propósito; guía para .env

Variables principales del proyecto (ver `.env.example`):

JUPYTER_TOKEN=token1234
PYSPARK_PYTHON=python3
SPARK_LOCAL_IP=0.0.0.0
PORT_JUPYTER=8888
PORT_SPARK=4040
PORT_POSTGRES=5432
PORT_WAREHOUSEUI=8080
POSTGRES_HOST=postgres_host
POSTGRES_DB=activo_financiero
POSTGRES_USER=usuario_spark
POSTGRES_PASSWORD=usuario_password
PGADMIN_DEFAULT_EMAIL=admin@admin.com
PGADMIN_DEFAULT_PASSWORD=adminPassword123
TICKERS= AAPL,MSFT,SPY,NVDA,INTC,AMD,GOOGL,AMZN,TSLA,JPM
START_DATE=01-01-2020
END_DATE=30-11-2025
RAW_TABLE=prices_daily
ANALYTICS_TABLE=daily_features
API_PORT=7000

Notas:
- La API usa por defecto el camino del contenedor `/models/best_pipeline.joblib`. Compose monta `MODEL_DIR` del host en `/models` para que el archivo esté disponible.
- Se puede ajustar `API_PORT` si se desea exponer otro puerto local.

#Guía de variables de entorno (.env)

- `JUPYTER_TOKEN`: token para acceso a Jupyter dentro del contenedor.
- `PYSPARK_PYTHON`: intérprete Python que usará PySpark (`python3`).
- `SPARK_LOCAL_IP`: IP local de Spark para UI y jobs (`0.0.0.0`).
- `PORT_JUPYTER`: puerto de Jupyter Notebook (por ejemplo `8888`).
- `PORT_SPARK`: puerto de Spark UI (por ejemplo `4040`).
- `PORT_POSTGRES`: puerto del servicio Postgres (por ejemplo `5432`).
- `PORT_WAREHOUSEUI`: puerto de UI/pgAdmin (por ejemplo `8080`).
- `POSTGRES_HOST`: hostname del contenedor/database para conexión (`postgres_host`).
- `POSTGRES_DB`: nombre de la base de datos (`activo_financiero`).
- `POSTGRES_USER`: usuario de conexión (ejemplo `usuario_spark`).
- `POSTGRES_PASSWORD`: contraseña del usuario de conexión.
- `PGADMIN_DEFAULT_EMAIL`: email por ejemplo de pgAdmin.
- `PGADMIN_DEFAULT_PASSWORD`: contraseña por ejemplo de pgAdmin.
- `TICKERS`: lista separada por comas de símbolos a descargar (ej. `AAPL,MSFT,SPY,...`).
- `START_DATE`: fecha inicial (formato `DD-MM-YYYY`, ej. `01-01-2020`).
- `END_DATE`: fecha final (formato `DD-MM-YYYY`, ej. `30-11-2025`).
- `RAW_TABLE`: nombre de tabla RAW para precios diarios (`prices_daily`).
- `ANALYTICS_TABLE`: nombre de tabla de features derivadas (`daily_features`).
- `API_PORT`: puerto local para la API de FastAPI (por ejemplo `7000`).

#Arquitectura

 Inicialización de esquemas raw y analytics
        │
        ▼
 Notebook yf_ingesta.ipynb para ingesta de datos hacia RAW
        │
        ▼
 Script build_features.py para construcción de tabla hacia ANALYTICS
        │
        ▼
 Preparación y Entrenamiento (EDA + balanceo + pipeline)
        │
        ▼
 Obtención del mejor modelo (`joblib`) y simulación de trading
        │
        ▼
 Serving con FastAPI del mejor modelo bajo endpoint predict

#Estructura de directorios

📁 init-scripts             -> SQL inicial y esquemas
│   └── 01-init-schemas.sql
📁 libros                   -> Notebooks y artefactos del modelo de ingesta + ML
│   ├── yf_ingesta.ipynb
│   ├── ml_trading_classifier.ipynb
│   └── modelos/            -> Artefacto del modelo (`best_pipeline.joblib`)
📁 model-api                -> Código de la API FastAPI (serving)
📁 models                   -> Carpeta auxiliar de modelos (si aplica)
📁 scripts                  -> Scripts utilitarios (ETL/Features/OBT si aplica)
│   └── build_features.py
📁 warehouse_data           -> Datos persistidos (volumenes/BD)
📁 warehouse_ui_data        -> Datos de UI (pgAdmin/sesiones)
docker-compose.yaml         -> Orquestación de servicios (API, montajes)
Dockerfile.feature-builder  -> Dockerfile para construir/servir la API
README.md                   -> Documentación del proyecto
requirements.txt            -> Dependencias Python

#Pasos de ejecución (end-to-end)

Prerrequisitos
* Docker y Docker Compose instalados
* Archivo `.env` configurado con variables de datos y puertos

1) Clonar y configurar

- Clonar el repositorio:
git clone https://github.com/johnnyredwood/Proyecto-ActivoFinanciero


- Crear `.env` desde plantilla y ajustar valores:
cp .env.example .env


2) Levantar infraestructura base

- En la raíz del proyecto, levantar contenedores base:
docker-compose --env-file .env up -d
docker-compose ps


3) Ingesta de datos RAW (Yahoo Finance)

- Ejecutar el notebook `libros/yf_ingesta.ipynb` dentro de Jupyter para poblar la tabla RAW (`RAW_TABLE`, por defecto `prices_daily`).
- Acceso a Jupyter (si está habilitado): `http://localhost:${PORT_JUPYTER}` con token `${JUPYTER_TOKEN}`.

4) Construcción de tabla ANALYTICS (features)

- Ejecutar el constructor de features con parámetros del `.env` se puede utilizar el modo full o modo 

docker compose run --rm 
       -e RAW_TABLE=prices_daily 
       -e ANALYTICS_TABLE=daily_features 
       feature-builder 
       /app/scripts/build_features.py --mode full 
       --ticker AAPL,MSFT,SPY,NVDA,INTC,AMD,GOOGL,AMZN,TSLA,JPM 
       --start-date 2020-01-01 
       --end-date 2025-11-30 
       --run-id full_load 
       --overwrite true 
       --vol-window 20

Parámetros del constructor de features (`build_features.py`):
- `--mode` (obligatorio):
       - `full`: procesa el rango completo definido por `--start-date` y `--end-date` para todos los tickers provistos.
       - `by-date-range`: procesa exactamente el rango indicado (útil para cargas parciales o incrementales).
- `--ticker` (obligatorio): lista separada por comas de símbolos. Ej: `AAPL,MSFT,SPY`.
- `--start-date` (obligatorio): fecha inicial en formato `YYYY-MM-DD`. Ej: `2020-01-01`.
- `--end-date` (obligatorio): fecha final en formato `YYYY-MM-DD`. Ej: `2025-11-30`.
- `--run-id` (obligatorio): identificador de ejecución para trazabilidad (ej. `full_load`, `daily_job_2025_11_30`).
- `--overwrite` (opcional, `true|false`, por defecto `false`):
       - `true`: reescribe particiones/registros existentes del rango/tickers indicados.
       - `false`: preserva registros existentes y solo inserta nuevos.
- `--vol-window` (opcional, entero, por defecto `20`): ventana en días para cálculo de volatilidad y estadísticas móviles.

Variables de ambiente utilizadas por el servicio:
- `RAW_TABLE`: nombre de la tabla RAW desde la que se leen precios (ej. `prices_daily`).
- `ANALYTICS_TABLE`: nombre de la tabla destino de features derivadas (ej. `daily_features`).

5) Entrenamiento y selección de modelo

- Ejecutar el notebook `libros/ml_trading_classifier.ipynb` usando como fuente la tabla del esquema ANALYTICS.
- Se Exportará el mejor pipeline a `libros/modelos/best_pipeline.joblib`.

6) Construir y levantar la API del modelo

- Construir la imagen del servicio `model-api` y levantarlo:

docker compose build model-api
docker compose up -d model-api
docker compose ps

7) Probar la API

- Endpoint Health con ejemplo de llamada:

Invoke-RestMethod -Uri http://localhost:{API_PORT}/health

-UI para la API: abrir `http://localhost:{API_PORT}/docs` en el navegador.

- Endpoint Predict con ejemplo de llamada usando mis features seleccionadas:
Invoke-RestMethod -Method Post -Uri http://localhost:{API_PORT}/predict -ContentType 'application/json' -Body '{
       "year": 2025,
       "month": 6,
       "day_of_week": 3,
       "open": 150.23,
       "volume": 1234567,
       "return_prev_close": 0.0045,
       "volatility_n_days": 0.012,
       "is_monday": false,
       "is_friday": true
}'

Respuesta esperada:
{"pred_label": 1}

EDA y Modelado (resumen):

- EDA breve: distribución de retornos (histogramas), balance de clases global y por ticker, rango temporal y shape del dataset.
- Balanceo de clases en Train: opción de undersampling de la mayoría u oversampling de la minoría; se utiliza `balanced_train_df` para entrenar.
- Pipeline: preprocesamiento de numéricas y categóricas, modelo de clasificación (e.g., XGBoost/LightGBM/RandomForest), selección por métricas en validación.
- Exportación: `joblib.dump(pipeline, 'libros/modelos/best_pipeline.joblib')`.
- Features utilizados para evitar leakage: "year", "month", "day_of_week","open", "return_prev_close", "volatility_n_days", "volume","is_monday", "is_friday"
- Modelos utilizados para buscar mejor predictor (clasificación binaria): Regresión Logística, Decision Tree, Random Forest, Gradient Boosting, AdaBoosting, XGBoosting, Light Gradient Boosting con eso se pudo decidir el mejor modelo de entre esos que para el caso de mis datos fue XGBoosting
- En el presente caso de mi proyecto se utiliza para el entrenamiento datos del año anterior a 2023, para validación datos del 2024 y para testing datos del 2025

#Simulación 2025 (backtest)

Regla simple:
- Si `pred_label == 1`, comprar al open y cerrar al close del día; si `0`, estar en efectivo.

Outputs:
- Curva de equity del activo y portafolio (si varios tickers).
- Drawdown máximo, retorno total y anualizado, número de trades.
- Comparación con métricas ML (accuracy, precision/recall si aplica).

#Troubleshooting

- La API muestra `prob_up`: reconstruir la imagen y reiniciar Compose; hacer hard-refresh de `/docs` (Ctrl+F5). La respuesta final solo incluye `pred_label`.
- Error cargando modelo: confirmar que `libros/modelos` está montado en `/models` y que el archivo existe como `best_pipeline.joblib`.
- Puerto ocupado: cambiar `API_PORT` en `.env` y actualizar el mapeo en `docker-compose.yaml`.

#Conclusiones generales:

- Enfoque: predecimos si el día cerrará arriba del precio de apertura (`target_up`). Usamos solo datos disponibles antes de abrir el mercado para evitar errores por usar información futura.
- EDA: los datos son estables en el tiempo y hay días al alza y a la baja en proporciones razonables. Las estadísticas muestran datos con señales muy fluctuantes lo cual es lógico en el sentido de que en el mercado hay continuos movimientos de alzas y bajas
- Features: usamos retornos previos, volatilidad calculada con ventanas, volumen y datos del calendario (día de la semana, mes). Todo se calcula con información que se puede tener disponible al inicio del día esto para evitar leakage.
- Splits: entrenamos con 2020–2023, validamos con 2024 y probamos con 2025 para medir qué tanto generaliza el modelo.
- Modelos: probamos varios y XGBoost fue el mejor, con buenas métricas (F1 y ROC-AUC) y comportamiento más estable.
- 2025: el modelo mantiene buen rendimiento fuera de muestra y sirve para una estrategia sencilla basada en la señal al inicio del día.
- Valor: el pipeline es reproducible y útil para decisiones tácticas de corto plazo; no reemplaza análisis profundo del mercado ni expertise de traders.
- Futuro: mejorar el tuning, añadir features de microestructura, sentiment analysis, probar la idea en más activos y con datos de más años.