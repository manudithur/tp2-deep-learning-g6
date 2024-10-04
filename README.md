
# Nutritional Plan RAG

Este repositorio contiene un trabajo práctico realizado para el Instituto Tecnológico de Buenos Aires (ITBA) en la materia 73.64 - Temas Avanzados en Deep Learning.

## Requisitos previos

Antes de utilizar el proyecto, asegúrate de tener instalados los requisitos especificados en `requirements.txt`. Puedes hacerlo ejecutando el siguiente comando en tu entorno:

```bash
pip install -r requirements.txt
```

Además, necesitarás configurar un archivo `.env` con las siguientes variables de entorno:

- `HUGGINGFACEHUB_API_TOKEN`: Token de acceso a la API de Hugging Face.
- `DATA_PATH`: Ruta al archivo CSV que contiene los datos de los planes base.
- `EMBEDDING_MODEL_NAME`: Nombre del modelo de embeddings que se utilizará.
- `OPENAI_API_KEY`: Clave de la API de OpenAI para generar respuestas a través de un LLM.

## Uso del programa

Este repositorio está estructurado alrededor de una clase principal encargada de la orquestación del vector store y el modelo de lenguaje. El propósito del proyecto es generar planes nutricionales personalizados en función del perfil del usuario y permitir preguntas específicas sobre los planes.

### Clases principales

#### LLM.py

Esta clase es la encargada de coordinar la interacción entre el almacenamiento de vectores (vector store) y el LLM. Posee dos funciones principales:

##### 1. `Query NutriSense`
Genera un plan de comidas personalizado basado en el perfil del usuario.

- **Input**: Un perfil del usuario en el siguiente formato:

  ```python
  profile = {
      "age": 45,
      "gender": "male" | "female",
      "height": 180,
      "weight": 85,
      "activity_level": "Sedentary" | "Lightly Active" | "Moderately Active" | "Very Active",
      "fitness_goal": "Weight Loss" | "Muscle Gain" | "Maintenance",
      "diet_type": "Omnivore" | "Vegetarian" | "Vegan",
      "tastes": ["taste1", "taste2", ... ]
  }
  ```

- **Output**: Devuelve un plan de comidas personalizado, el prompt generado, documentos del contexto utilizados y un plan base de referencia.

##### 2. `Query question`
Permite realizar una pregunta relacionada con nutrición o los planes alimenticios.

- **Input**: Una pregunta en formato de texto.
- **Output**: Devuelve la respuesta del modelo LLM, el prompt generado y los documentos de contexto relevantes.

#### Metrics.py

Esta clase permite realizar un análisis y evaluación de las métricas relacionadas con los resultados obtenidos por el modelo.

- **Inicialización de la clase**: Para utilizar las métricas, debes inicializar la clase `Metrics` de la siguiente manera:

  ```python
  metrics = Metrics(prompt, actual_output, retrieved_docs, expected_output=None)
  ```

- **Métodos disponibles**: Llama a los métodos correspondientes para evaluar el rendimiento del modelo.

## Ejemplo de uso

### Generar un plan nutricional

```python
from LLM import LLM

# Crear un perfil de usuario
profile = {
    "age": 30,
    "gender": "female",
    "height": 165,
    "weight": 70,
    "activity_level": "Moderately Active",
    "fitness_goal": "Weight Loss",
    "diet_type": "Vegetarian",
    "tastes": ["spicy", "sweet"]
}

# Instanciar el modelo y generar un plan
llm = LLM()
plan, prompt, docs, base_plan = llm.query_nutrisense(profile)

# Imprimir resultados
print(plan)
```

### Hacer una pregunta específica

```python
from LLM import LLM

# Instanciar el modelo y hacer una pregunta
llm = LLM()
response, prompt, docs = llm.query_question("What is a good breakfast for weight loss?")

# Imprimir la respuesta
print(response)
```