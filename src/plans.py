import pandas as pd
from enum import Enum

class Plans:

    class GENDER(Enum):
        MALE = "male"
        FEMALE = "female"

    class ACTIVITY_LEVEL(Enum):
        SEDENTARY = "Sedentary"
        LIGHTLY_ACTIVE = "Lightly Active"
        MODERATELY_ACTIVE = "Moderately Active"
        VERY_ACTIVE = "Very Active"

    class FITNESS_GOAL(Enum):
        WEIGHT_LOSS = "Weight Loss"
        MUSCLE_GAIN = "Muscle Gain"
        MAINTENANCE = "Maintenance"

    class DIET_TYPE(Enum):
        OMNIVORE = "Omnivore"
        VEGETARIAN = "Vegetarian"
        VEGAN = "Vegan"

    def __init__(self):
        # Read the dataset
        self.df = pd.read_csv("hf://datasets/sarthak-wiz01/nutrition_dataset/nutrition_dataset.csv")

        # Convert relevant columns to numeric types (integers or floats)
        self.df['Age'] = pd.to_numeric(self.df['Age'], errors='coerce')
        self.df['Height'] = pd.to_numeric(self.df['Height'], errors='coerce')
        self.df['Weight'] = pd.to_numeric(self.df['Weight'], errors='coerce')

    def get_plan(self, age, gender, height, weight, activity_level, fitness_goal, diet_type):
        # Filter the dataset based on user input
        filtered_df = self.df[
            (self.df['Age'] >= age - 3) & (self.df['Age'] <= age + 3) &
            (self.df['Gender'].str.lower() == gender.value) &  # Compare enum value
            (self.df['Height'] >= height - 5) & (self.df['Height'] <= height + 5) &
            (self.df['Weight'] >= weight - 5) & (self.df['Weight'] <= weight + 5) &
            (self.df['Activity Level'] == activity_level.value) &  # Compare enum value
            (self.df['Fitness Goal'] == fitness_goal.value) &  # Compare enum value
            (self.df['Dietary Preference'] == diet_type.value)  # Compare enum value
        ]

        if filtered_df.empty:
            raise ValueError("No plans found for the given input. Please adjust your criteria.")
        
        # Get the first plan from the filtered dataset
        plan = filtered_df.iloc[0]
        return plan
