
from openai import OpenAI

client = OpenAI()

def get_response_gpt4o(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for creating a weekly meal plan. You must provide breakfast, lunch, snack and dinner (with a dessert) options for each day of the week. The meal plan should meet the daily nutritional requirements for calories, fat, carbohydrates, and protein."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
