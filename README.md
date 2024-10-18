# Predicción de ACV - Aprendizaje de Máquina 1

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

## Environment setup
To create a Python virtual environment for the project:
```bash
python -m venv .venv
```

To activate the virtual environment:
```bash
source .venv/bin/activate
```

To install the required dependencies:
```bash
pip install -r requirements.txt
```

To download the dataset from Kaggle:
```bash
./download_dataset.sh
```