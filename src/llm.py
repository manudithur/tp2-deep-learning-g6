
from openai import OpenAI
from src.plans import Plans
from src.models import Profile, DIET_TYPE
from src.vector_store import VectorDB
import os
client = OpenAI()

class LLM:
    def __init__(self):
        data_path = os.getenv('DATA_PATH')
        embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
        # check if folder exists
        load_data = not os.path.exists('./src/chroma_db')
        self.vectorDB = VectorDB(data_path, embedding_model_name, load_data=load_data)

    def __get_response_gpt4o__(self, prompt):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for creating a weekly meal plan. You must provide breakfast, lunch, snack and dinner (with a dessert) options for each day of the week. The meal plan should meet the daily nutritional requirements for calories, fat, carbohydrates, and protein."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content
    
    def __format_docs__(self, docs):
        formatted_docs_string = ''

        for doc in docs:
            formatted_docs_string += f'Name: {doc.metadata["Name"]} \tDescription: {doc.page_content}\tCalories: {doc.metadata["Calories"]}\tCarbohydrates: {doc.metadata["Carbohydrates"]}\tFat: {doc.metadata["Fat"]}\tProtein: {doc.metadata["Protein"]}\n'
        
        return formatted_docs_string

    def query_nutrisense(self, user_data: Profile):
        print('-------------------------------------')
        print('Getting plan')
        plan = Plans().get_plan(user_data.age, user_data.gender, user_data.height, user_data.weight, user_data.activity_level, user_data.fitness_goal, user_data.diet_type)

        tastes_string = ''
        for taste in user_data.tastes:
            tastes_string += f'{taste} '

        restrictions = user_data.restrictions

        if user_data.diet_type == DIET_TYPE.VEGAN.value or user_data.diet_type == DIET_TYPE.VEGETARIAN.value:
            restrictions.append(user_data.diet_type.lower())

        print('Looking for documents')
        breakfast_docs = self.vectorDB.similarity_search(tastes_string, 20, {"Type": ["breakfast", "-"], "Restrictions": restrictions})
        snack_docs = self.vectorDB.similarity_search(tastes_string, 20, {"Type": ["snack", "-"], "Restrictions": restrictions})
        lunch_dinner_docs = self.vectorDB.similarity_search(tastes_string, 30, {"Type": ["lunch", "dinner"], "Restrictions": restrictions})
        dessert_docs = self.vectorDB.similarity_search(tastes_string, 20, {"Type": ["dessert", '-'], "Restrictions": restrictions})
        print('Got documents')
        
        prompt = f"""I need to create a weekly meal plan. Include meals from monday to sunday. 

        Must include breakfast, lunch, dinner with a dessert, and snack.

        The answers should be in the following format:

        Monday:
        Breakfast: name of the breakfast option
        Lunch: name of the lunch option
        Snack: name of the snack option
        Dinner: name of the dinner option
        Dessert: name of the dessert option

        Total calories: total calories for the day
        Total fat: total fat for the day
        Total carbohydrates: total carbohydrates for the day
        Total protein: total protein for the day

        Tuesday:
        ...
        

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
        {self.__format_docs__(breakfast_docs)}

        For Lunch and Dinner only include from the following options:
        {self.__format_docs__(lunch_dinner_docs)}

        For Snack only include from the following options:
        {self.__format_docs__(snack_docs)}

        For Dessert only include from the following options:
        {self.__format_docs__(dessert_docs)}

        """

        instructions = f"""
        Once you build the meal plan, calculate an estimate of the total calories, fat, carbohydrates, and protein for each day.
        Check for the difference between the daily nutritional requirements and the estimated values.

        Adjust servings or meals to get as close as possible to the daily nutritional requirements.
        
        Finally provide a average of the total calories, fat, carbohydrates, and protein for the whole week.
        """

        print('Built prompt with augmented context')

        response = self.__get_response_gpt4o__(prompt + context + instructions)

        print('Got response from GPT-4o')

        print("-------------------------------------")
        # response = ''

        retrieved_docs = []
        for doc in (breakfast_docs + snack_docs + lunch_dinner_docs + dessert_docs):
            retrieved_docs.append(self.__format_docs__([doc]))

        return response, prompt + instructions, retrieved_docs, Plans().format_plan(plan)

    def query_question(self, prompt):

        print("-------------------------------------")
        print('Looking for documents')

        docs = self.vectorDB.similarity_search(prompt, 5)

        print('Got documents')

        formatted_docs_string = self.__format_docs__(docs)

        prompt = f"""I need to answer the following question: {prompt}."""

        context = f"""Only answer using information from the following documents: {formatted_docs_string}"""

        print('Built prompt with augmented context')

        response = self.__get_response_gpt4o__(prompt + context)

        print('Got response from GPT-4o')

        print("-------------------------------------")

        retrieved_docs = []
        for doc in docs:
            retrieved_docs.append(self.__format_docs__([doc]))

        return response, prompt, retrieved_docs