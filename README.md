# Predicción de derrame cerebral - Aprendizaje de Máquina 1

Autores: María Fabiana Cid y Gonzalo Gabriel Fernandez, Universidad de Buenos Aires

## Objetivo

Según la Organización Mundial de la Salud (OMS), el accidente cerebrovascular (ACV) es la segunda causa principal de muerte a nivel mundial, siendo responsable de aproximadamente el 11% del total de muertes.

En el presente trabajo se analiza el conjunto de datos titulado [Stroke Prediction](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) publicado en Kaggle.

Se plantea como objetivo **predecir**, conociendo determinadas caracteristicas fisiológicas y sociales del paciente, la posibilidad de que tenga un accidente cardiovascular.

Primero se realiza la recolección de datos y su preparación para ser analizados. Luego, se realiza una ingenieria de features, el entrenamiento de distintos modelos y su evaluación.

De todos los algoritmos clásicos de aprendizaje de máquina estudiados, el de mejor desempeño fue el Random Forest con los siguiente indicadores:

- Accuracy: 0.964524
- Precisión: 0.961105
- Recall: 0.968041
- F1-score: 0.964561

Además, se entrenaron modelos de redes neuronales (aprendizaje profundo con las librerias Torch y Tensorflow). También se incorporaron herramientas de aprendizaje no supervisado, como Tsne para reducción de dimensionalidad y clustering con K-means.

## Instalación
Crear entorno virtual de Python para el proyecto:
```bash
python -m venv .venv
```

Activar el entorno virtual:
```bash
source .venv/bin/activate
```

Instalar las dependencias del proyecto:
```bash
pip install -r requirements.txt
```
Descargar el dataset de Kaggle:
```bash
./download_dataset.sh
```

## Información de los atributos del dataset utilizado

- id: identificador único
- gender: "Male" (Hombre), "Female" (Mujer) o "Other" (Otro)
- age: edad del paciente
- hypertension: 0 si el paciente no tiene hipertensión, 1 si el paciente tiene hipertensión
- heart_disease: 0 si el paciente no tiene enfermedades cardíacas, 1 si el paciente tiene una enfermedad cardíaca
- ever_married: "No" o "Yes" (No o Sí)
- work_type: "children" (niños), "Govt_job" (empleo gubernamental), "Never_worked" (nunca trabajó), "Private" (sector privado) o "Self-employed" (autónomo)
- Residence_type: "Rural" o "Urban" (Rural o Urbano)
- avg_glucose_level: nivel promedio de glucosa en sangre
- bmi: índice de masa corporal
- smoking_status: "formerly smoked" (fumó anteriormente), "never smoked" (nunca fumó), "smokes" (fuma) o "Unknown" (desconocido)
- stroke: 1 si el paciente sufrió un accidente cerebrovascular, 0 si no

Link to the [Stroke Detection dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/code).

# Puesta en producción del modelo entrenado - Aprendizaje de Máquina 2
## Instalación

1. Para poder levantar todos los servicios, primero instala [Docker](https://docs.docker.com/engine/install/) en tu 
computadora (o en el servidor que desees usar).

2. Clona este repositorio.

3. Crea las carpetas `airflow/config`, `airflow/dags`, `airflow/logs`, `airflow/plugins`, 
`airflow/logs`.

4. Si estás en Linux o MacOS, en el archivo `.env`, reemplaza `AIRFLOW_UID` por el de tu 
usuario o alguno que consideres oportuno (para encontrar el UID, usa el comando 
`id -u <username>`). De lo contrario, Airflow dejará sus carpetas internas como root y no 
podrás subir DAGs (en `airflow/dags`) o plugins, etc.

5. En la carpeta raíz de este repositorio, ejecuta:

```bash
docker compose --profile all up
```

6. Una vez que todos los servicios estén funcionando (verifica con el comando `docker ps -a` 
que todos los servicios estén healthy o revisa en Docker Desktop), podrás acceder a los 
diferentes servicios mediante:
   - Apache Airflow: http://localhost:8080
   - MLflow: http://localhost:5000
   - MinIO: http://localhost:9001 (ventana de administración de Buckets)
   - API: http://localhost:8800/
   - Documentación de la API: http://localhost:8800/docs

Si estás usando un servidor externo a tu computadora de trabajo, reemplaza `localhost` por su IP 
(puede ser una privada si tu servidor está en tu LAN o una IP pública si no; revisa firewalls 
u otras reglas que eviten las conexiones).

Todos los puertos u otras configuraciones se pueden modificar en el archivo `.env`. Se invita 
a jugar y romper para aprender; siempre puedes volver a clonar este repositorio.

## Apagar los servicios

Estos servicios ocupan cierta cantidad de memoria RAM y procesamiento, por lo que cuando no 
se están utilizando, se recomienda detenerlos. Para hacerlo, ejecuta el siguiente comando:

```bash
docker compose --profile all down
```

Si deseas no solo detenerlos, sino también eliminar toda la infraestructura (liberando espacio en disco), 
utiliza el siguiente comando:

```bash
docker compose down --rmi all --volumes
```

Nota: Si haces esto, perderás todo en los buckets y bases de datos.

## Simulación de servidor externo con dataset
Una de las tareas del servicio de airflow es realizar un fetch del datset.
Para eso simulamos un servidor externo que contiene el archivo zip con el dataset.

Una forma de simularlo es mediante un servidor http con python:

```sh
python -m http.server -d data/ 12000
```

En este caso el puerto utilizado es el 12000 y se sirve la carpeta `data/`.
Luego, se debe actualizar la URL al archivo zip en la configuración del DAG (tener en cuenta que se debe utilizar la
dirección IP del sistema en la red local para que los servicios de docker puedan direccionarlo, localhost no funcionará).

IMPORTANTE: Es necesario actualizar en el archivo `etl_process.py` la dirección IP para descarga del ZIP con
la correspondiente a la máquina simulando el servidor.
