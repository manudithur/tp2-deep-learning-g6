import pandas as pd
from enum import Enum

class Plans:
    def __init__(self):
        # Read the dataset
        self.df = pd.read_csv("hf://datasets/sarthak-wiz01/nutrition_dataset/nutrition_dataset.csv")

        # Convert relevant columns to numeric types (integers or floats)
        self.df['Age'] = pd.to_numeric(self.df['Age'], errors='coerce')
        self.df['Height'] = pd.to_numeric(self.df['Height'], errors='coerce')
        self.df['Weight'] = pd.to_numeric(self.df['Weight'], errors='coerce')

    def get_plan(self, age, gender, height, weight, activity_level, fitness_goal, diet_type):
        # Add a new column 'Score' to track the matching score for each plan
        self.df['Score'] = 0

        # Exact match conditions
        filtered_df = self.df.copy()
        filtered_df = filtered_df[
            (filtered_df['Gender'].str.lower() == gender)  # Exact gender match
        ]

        # Age matching with a range of +/- 3 years
        filtered_df['Score'] += (filtered_df['Age'] >= age - 3) & (filtered_df['Age'] <= age + 3)

        # Height matching with a range of +/- 5 cm
        filtered_df['Score'] += (filtered_df['Height'] >= height - 5) & (filtered_df['Height'] <= height + 5)

        # Weight matching with a range of +/- 5 kg
        filtered_df['Score'] += (filtered_df['Weight'] >= weight - 5) & (filtered_df['Weight'] <= weight + 5)

        # Activity level match
        filtered_df['Score'] += (filtered_df['Activity Level'] == activity_level)

        # Fitness goal match
        filtered_df['Score'] += (filtered_df['Fitness Goal'] == fitness_goal)

        # Diet type match
        filtered_df['Score'] += (filtered_df['Dietary Preference'] == diet_type)

        # Sort by the highest score
        sorted_df = filtered_df.sort_values(by='Score', ascending=False)
        
        # If no matching plan is found with a score > 0
        if sorted_df.empty or sorted_df['Score'].max() == 0:
            raise ValueError("No suitable plans found for the given input. Please adjust your criteria.")

        # Get the best-matching plan (highest score)
        best_plan = sorted_df.iloc[0]
        return best_plan

