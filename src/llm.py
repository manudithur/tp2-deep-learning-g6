
from openai import OpenAI
from src.plans import Plans
from src.models import Profile, DIET_TYPE
from src.vector_store import VectorDB
import os
client = OpenAI()

def __get_response_gpt4o__(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for creating a weekly meal plan. You must provide breakfast, lunch, snack and dinner (with a dessert) options for each day of the week. The meal plan should meet the daily nutritional requirements for calories, fat, carbohydrates, and protein."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def query_nutrisense(user_data: Profile):
    print("Getting plan")
    plan = Plans().get_plan(user_data.age, user_data.gender, user_data.height, user_data.weight, user_data.activity_level, user_data.fitness_goal, user_data.diet_type)
    print("Got plan: ", plan)

    data_path = os.getenv('DATA_PATH')
    embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
    # check if folder exists
    load_data = not os.path.exists('./src/chroma_db')
    vectorDB = VectorDB(data_path, embedding_model_name, load_data=load_data)

    tastes_string = ''
    for taste in user_data.tastes:
        tastes_string += f'{taste} '

    restrictions = user_data.restrictions

    if user_data.diet_type == DIET_TYPE.VEGAN.value or user_data.diet_type == DIET_TYPE.VEGETARIAN.value:
        restrictions.append(user_data.diet_type.lower())

    print("Tastes: ", tastes_string)
    print("Restrictions: ", restrictions)

    print('-------------------------------------')
    print('Looking for documents')
    breakfast_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["breakfast", "-"], "Restrictions": restrictions})
    snack_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["snack", "-"], "Restrictions": restrictions})
    lunch_dinner_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["lunch", "dinner"], "Restrictions": restrictions})
    dessert_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["dessert", '-'], "Restrictions": restrictions})
    specific_docs = vectorDB.similarity_search(tastes_string, 30, {"Type": ["specific_snacks", '-'], "Restrictions": restrictions})

    def format_docs(docs):
        formatted_docs_string = ''

        for doc in docs:
            formatted_docs_string += f'Meal: {doc.page_content}\tCalories: {doc.metadata["Calories"]}\tCarbohydrates: {doc.metadata["Carbohydrates"]}\tFat: {doc.metadata["Fat"]}\tProtein: {doc.metadata["Protein"]}\n'
        
        return formatted_docs_string

    prompt = f"""I need to create a weekly meal plan. Include meals from monday to sunday. 

    Must include breakfast, lunch, dinner with a dessert, and snack.

    The plan needs to comply with the following daily nutritional requirements:
    Calories: {plan['Daily Calorie Target']}
    Fat: {plan['Fat']}
    Carbohydrates: {plan['Carbohydrates']}
    Protein: {plan['Protein']}

    Try to get as close as possible to this values.

    I have the following tastes: {', '.join(user_data.tastes)}
    This is my dietary preference: {user_data.diet_type}

    What are some meal options for breakfast, lunch, and dinner that I can include in my meal plan?
    """
    context = f"""

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

    print("CONTEXT:\n", context)

    response = __get_response_gpt4o__(prompt + context)

    retrieved_docs_names = []

    return response, prompt, retrieved_docs_names