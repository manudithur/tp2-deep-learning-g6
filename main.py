from src.vector_store import VectorDB
import  os
from dotenv import load_dotenv, find_dotenv
from src.plans import Plans
from src.llm import get_response_gpt4o
import requests
_ = load_dotenv(find_dotenv())

user_data = {
    "age": 25,
    "gender": Plans.GENDER.MALE,
    "height": 180,
    "weight": 80,
    "activity_level": Plans.ACTIVITY_LEVEL.MODERATELY_ACTIVE,
    "fitness_goal": Plans.FITNESS_GOAL.WEIGHT_LOSS,
    "diet_type": Plans.DIET_TYPE.OMNIVORE,
    "tastes": ["fish", "beef", "salad", "argentinian", "waffles", "eggs", "cheese"]
}

plan = Plans().get_plan(user_data["age"], user_data["gender"], user_data["height"], user_data["weight"], user_data["activity_level"], user_data["fitness_goal"], user_data["diet_type"])

data_path = os.getenv('DATA_PATH')
embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
vectorDB = VectorDB(data_path, embedding_model_name, load_data=True)

tastes_string = ''
for taste in user_data['tastes']:
    tastes_string += f'{taste} '


breakfast_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["breakfast", "-"]})
snack_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["snack", "-"]})
lunch_dinner_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["lunch", "dinner"]})
dessert_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["dessert", '-']})

def format_docs(docs):
    formatted_docs_string = ''

    for doc in docs:
        formatted_docs_string += f'Meal: {doc.page_content}\tCalories: {doc.metadata["Calories"]}\tCarbohydrates: {doc.metadata["Carbohydrates"]}\tFat: {doc.metadata["Fat"]}\tProtein: {doc.metadata["Protein"]}\n'
    
    return formatted_docs_string

print('------------------------------------------')
prompt = f"""I need to create a weekly meal plan. Include meals from monday to sunday. 

Must include breakfast, lunch, dinner with a dessert, and snack.

The plan needs to comply with the following daily nutritional requirements:
Calories: {plan['Daily Calorie Target']}
Fat: {plan['Fat']}
Carbohydrates: {plan['Carbohydrates']}
Protein: {plan['Protein']}

Try to get as close as possible to this values.

I have the following tastes: {', '.join(user_data['tastes'])}

What are some meal options for breakfast, lunch, and dinner that I can include in my meal plan?

For Breakfast only include from the following options:
{format_docs(breakfast_docs)}

For Lunch and Dinner only include from the following options:
{format_docs(lunch_dinner_docs)}

For Snack only include from the following options:
{format_docs(snack_docs)}

For Dessert only include from the following options:
{format_docs(dessert_docs)}

Adjust servings as needed to meet the daily nutritional requirements.
If you cant build a meal plan with this meals, feel free to refuse the request.

Once you choose a meal plan, adjust the servings as needed to meet as close as possible the daily nutritional requirements.
"""

print(prompt)
print('------------------------------------------')


response = get_response_gpt4o(prompt)

print(response)