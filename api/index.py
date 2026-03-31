from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import os
import warnings
from sklearn.exceptions import DataConversionWarning

# Remove warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

# Fix paths for Vercel
BASE_DIR = os.path.dirname(__file__)

app = Flask(__name__, template_folder="../templates")

# Load dataset
data_path = os.path.join(BASE_DIR, "../fitness_data.csv")
data = pd.read_csv(data_path)

# 🔥 IMPORTANT: Reduce plan categories (fix ML issue)
def simplify_plan(plan):
    plan = plan.lower()
    if "loss" in plan or "cardio" in plan or "fat" in plan:
        return "weight_loss"
    elif "gain" in plan or "muscle" in plan or "strength" in plan:
        return "weight_gain"
    else:
        return "maintenance"

data['plan'] = data['plan'].apply(simplify_plan)

# Encoding
le_goal = LabelEncoder()
le_category = LabelEncoder()
le_plan = LabelEncoder()

data['goal'] = le_goal.fit_transform(data['goal'])
data['category'] = le_category.fit_transform(data['category'])
data['plan'] = le_plan.fit_transform(data['plan'])

# Features
X = data[['weight', 'height', 'age', 'goal', 'BMI']]
y = data['plan']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# ---------------- FUNCTIONS ---------------- #

def calculate_bmi(weight, height):
    return round(weight / ((height / 100) ** 2), 2)

def get_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    else:
        return "Overweight"

def ideal_weight(height):
    return round((height - 100) * 0.9, 2)

def get_progress(weight, ideal):
    diff = round(weight - ideal, 2)
    if diff > 0:
        return f"You need to lose {diff} kg"
    elif diff < 0:
        return f"You need to gain {abs(diff)} kg"
    else:
        return "You are at ideal weight"

# Convert ML output to readable text
def format_plan(plan):
    if plan == "weight_loss":
        return "Cardio + Fat loss workout"
    elif plan == "weight_gain":
        return "Strength training + Muscle gain"
    else:
        return "Balanced fitness routine"

def get_workout_plan(category, goal):
    if category == "Overweight":
        return "30 min walking + cardio + light strength training"
    elif category == "Normal":
        if goal == "weight loss":
            return "HIIT + cardio + abs workout"
        elif goal == "weight gain":
            return "strength training (push/pull/legs)"
        else:
            return "moderate exercise + yoga"
    else:
        return "strength training + low cardio + muscle gain workout"

def get_diet_plan(category, goal):
    if category == "Overweight":
        return "Low calorie diet, vegetables, high protein, avoid sugar"
    elif category == "Normal":
        if goal == "weight loss":
            return "Calorie deficit + protein + fiber foods"
        elif goal == "weight gain":
            return "High calorie + protein + healthy fats"
        else:
            return "Balanced diet (carbs + protein + fats)"
    else:
        return "High calorie diet, milk, nuts, banana, protein foods"

def get_tip(goal):
    if goal == "weight loss":
        return "Stay consistent and track calories"
    elif goal == "weight gain":
        return "Eat more and do strength training"
    else:
        return "Maintain balance and stay active"

def get_weekly_workout(goal):
    if goal == "weight loss":
        return {
            "Monday": "Cardio + Abs",
            "Tuesday": "HIIT",
            "Wednesday": "Rest",
            "Thursday": "Cardio + Core",
            "Friday": "Full Body",
            "Saturday": "Yoga",
            "Sunday": "Rest"
        }
    elif goal == "weight gain":
        return {
            "Monday": "Chest",
            "Tuesday": "Back",
            "Wednesday": "Legs",
            "Thursday": "Shoulders",
            "Friday": "Arms",
            "Saturday": "Light Cardio",
            "Sunday": "Rest"
        }
    else:
        return {
            "Monday": "Yoga",
            "Tuesday": "Cardio",
            "Wednesday": "Rest",
            "Thursday": "Strength",
            "Friday": "Walking",
            "Saturday": "Stretching",
            "Sunday": "Rest"
        }

def get_weekly_diet(goal):
    return {
        "Breakfast": "Oats + Milk + Fruits",
        "Lunch": "Rice + Dal + Vegetables",
        "Dinner": "Chapati + Paneer/Chicken",
        "Snacks": "Nuts + Banana"
    }

# ---------------- ROUTE ---------------- #

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        try:
            weight = float(request.form["weight"])
            height = float(request.form["height"])
            age = int(request.form["age"])
            goal = request.form["goal"].strip().lower()

            goal_encoded = le_goal.transform([goal])[0]

            bmi = calculate_bmi(weight, height)
            category = get_category(bmi)

            input_data = pd.DataFrame(
                [[weight, height, age, goal_encoded, bmi]],
                columns=['weight', 'height', 'age', 'goal', 'BMI']
            )

            prediction = model.predict(input_data)
            plan_raw = le_plan.inverse_transform(prediction)[0]
            plan = format_plan(plan_raw)

            workout = get_workout_plan(category, goal)
            diet = get_diet_plan(category, goal)
            tip = get_tip(goal)

            ideal = ideal_weight(height)
            progress = get_progress(weight, ideal)

            weekly_workout = get_weekly_workout(goal)
            weekly_diet = get_weekly_diet(goal)

            result = {
                "bmi": bmi,
                "category": category,
                "plan": plan,
                "workout": workout,
                "diet": diet,
                "tip": tip,
                "ideal": ideal,
                "progress": progress,
                "weekly_workout": weekly_workout,
                "weekly_diet": weekly_diet
            }

        except Exception as e:
            result = {
                "bmi": "Error",
                "category": "Invalid input",
                "plan": str(e)
            }

    return render_template("index.html", result=result)

# Vercel config
app.debug = False

# Local run
if __name__ == "__main__":
    app.run(debug=True)