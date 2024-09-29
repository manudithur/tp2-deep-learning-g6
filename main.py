from src.vector_store import VectorDB
import  os
from dotenv import load_dotenv, find_dotenv
from src.plans import Plans
from src.llm import get_response_gpt4o
import requests
_ = load_dotenv(find_dotenv())

user_data = {
    "age": 22,
    "gender": Plans.GENDER.MALE.value,
    "height": 188,
    "weight": 70,
    "activity_level": Plans.ACTIVITY_LEVEL.MODERATELY_ACTIVE.value,
    "fitness_goal": Plans.FITNESS_GOAL.MUSCLE_GAIN.value,
    "diet_type": Plans.DIET_TYPE.OMNIVORE.value,
    "restrictions": [Plans.RESTRICTIONS.GLUTEN_FREE.value],
    "tastes": ["fish", "beef", "salad", "indian", "waffles", "eggs", "cheese"]
}

plan = Plans().get_plan(user_data["age"], user_data["gender"], user_data["height"], user_data["weight"], user_data["activity_level"], user_data["fitness_goal"], user_data["diet_type"])

data_path = os.getenv('DATA_PATH')
embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
vectorDB = VectorDB(data_path, embedding_model_name)

tastes_string = ''
for taste in user_data['tastes']:
    tastes_string += f'{taste} '

restrictions = user_data['restrictions']

if user_data['diet_type'] == Plans.DIET_TYPE.VEGAN.value or user_data['diet_type'] == Plans.DIET_TYPE.VEGETARIAN.value:
    restrictions.append(user_data['diet_type'].lower())


breakfast_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["breakfast", "-"], "Restrictions": restrictions})
snack_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["snack", "-"], "Restrictions": restrictions})
lunch_dinner_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["lunch", "dinner"], "Restrictions": restrictions})
dessert_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["dessert", '-'], "Restrictions": restrictions})
specific_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["specific_snacks", '-'], "Restrictions": restrictions})

for doc in breakfast_docs:
    print(doc.metadata)

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
This is my dietary preference: {user_data['diet_type']}

What are some meal options for breakfast, lunch, and dinner that I can include in my meal plan?

For Breakfast only include from the following options:
{format_docs(breakfast_docs)}

For Lunch and Dinner only include from the following options:
{format_docs(lunch_dinner_docs)}

For Snack only include from the following options:
{format_docs(snack_docs)}

For Dessert only include from the following options:
{format_docs(dessert_docs)}


Once you build the meal plan, calculate an estimate of the total calories, fat, carbohydrates, and protein for each day.
Check for the difference between the daily nutritional requirements and the estimated values.

Then specify some specific snacks that can be added to meet the daily nutritional requirements.
Here you have things to eat on the go in order to meet the daily nutritional requirements:
{format_docs(specific_docs)}

Finally provide a average of the total calories, fat, carbohydrates, and protein for the whole week.

"""

print(prompt)
print('------------------------------------------')


response = get_response_gpt4o(prompt)

print(response)