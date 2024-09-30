
from dotenv import load_dotenv, find_dotenv
from src.plans import Plans
from src.models import Profile, DIET_TYPE, GENDER, ACTIVITY_LEVEL, FITNESS_GOAL, RESTRICTIONS
from src.llm import query_nutrisense
import pandas as pd

_ = load_dotenv(find_dotenv())

user_data = {
    "age": 22,
    "gender": GENDER.MALE.value,
    "height": 188,
    "weight": 70,
    "activity_level": ACTIVITY_LEVEL.MODERATELY_ACTIVE.value,
    "fitness_goal": FITNESS_GOAL.MUSCLE_GAIN.value,
    "diet_type": DIET_TYPE.OMNIVORE.value,
    "restrictions": [],
    "tastes": ["fish", "beef", "salad", "indian", "waffles", "eggs", "cheese"]
}

profile = Profile(**user_data)


response, prompt, retrieved_docs = query_nutrisense(profile)

print(response)
