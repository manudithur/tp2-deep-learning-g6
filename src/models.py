from enum import Enum
from typing import List

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

class RESTRICTIONS(Enum):
    GLUTEN_FREE = "celiac"
    DAIRY_FREE = "lactose-intolerant"

class Profile:
    def __init__(self, age: int, gender: GENDER, height: int, weight: int, 
                 activity_level: ACTIVITY_LEVEL, fitness_goal: FITNESS_GOAL, 
                 diet_type: DIET_TYPE, restrictions: List[RESTRICTIONS], tastes: List[str]):
        self.age = age
        self.gender = gender
        self.height = height
        self.weight = weight
        self.activity_level = activity_level
        self.fitness_goal = fitness_goal
        self.diet_type = diet_type
        self.restrictions = restrictions
        self.tastes = tastes