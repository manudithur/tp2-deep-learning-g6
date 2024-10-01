
from dotenv import load_dotenv, find_dotenv
from src.models import Profile, DIET_TYPE, GENDER, ACTIVITY_LEVEL, FITNESS_GOAL, RESTRICTIONS
from src.llm import LLM
from src.metrics import Metrics
_ = load_dotenv(find_dotenv())

llm = LLM()

user_data = {
    "age": 22,
    "gender": GENDER.MALE.value,
    "height": 188,
    "weight": 85,
    "activity_level": ACTIVITY_LEVEL.MODERATELY_ACTIVE.value,
    "fitness_goal": FITNESS_GOAL.MUSCLE_GAIN.value,
    "diet_type": DIET_TYPE.OMNIVORE.value,
    "restrictions": [RESTRICTIONS.DAIRY_FREE.value],
    "tastes": ["fish", "beef", "salad", "argentinian", "waffles", "eggs", "cheese"]
}

profile = Profile(**user_data)


response, prompt, retrieved_docs = llm.query_nutrisense(profile)

# print(response)

#response, prompt, retrieved_docs = llm.query_question("What is the nutritional value of a banana?")
print(prompt+"\n")
print(retrieved_docs)
print(response+"\n")

expected_response = "A banana contains 300 calories, 0.4 grams of fat, 27 carbohydrates, and 1.2 grams of protein."

metrics = Metrics(prompt, response, retrieved_docs, expected_response)

metrics.evaluate_answer_relevancy()
metrics.evaluate_faithfulness()

