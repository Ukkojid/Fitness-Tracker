from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Fix paths for Vercel
BASE_DIR = os.path.dirname(__file__)

app = Flask(__name__, template_folder="../templates")

# Load dataset safely
data_path = os.path.join(BASE_DIR, "../fitness_data.csv")
data = pd.read_csv(data_path)

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

    else:  # Underweight
        return "strength training + low cardio + muscle gain workout"

def get_diet_plan(category, goal):
    if category == "Overweight":
        return "Low calorie diet, more vegetables, high protein, avoid sugar"

    elif category == "Normal":
        if goal == "weight loss":
            return "Calorie deficit + protein + fiber rich foods"
        elif goal == "weight gain":
            return "High calorie + protein + healthy fats"
        else:
            return "Balanced diet (carbs + protein + fats)"

    else:  # Underweight
        return "High calorie diet, milk, nuts, banana, protein rich foods"

def get_tip(goal):
    if goal == "weight loss":
        return "Stay consistent and track calories"
    elif goal == "weight gain":
        return "Eat more frequently and lift weights"
    else:
        return "Maintain balance and stay active"

# ---------------- ROUTE ---------------- #

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        try:
            weight = float(request.form["weight"])
            height = float(request.form["height"])
            age = int(request.form["age"])
            goal = request.form["goal"].strip().lower()   # ✅ FIXED

            # Encode goal safely
            goal_encoded = le_goal.transform([goal])[0]

            # Calculate BMI & category
            bmi = calculate_bmi(weight, height)
            category = get_category(bmi)

            # ML Prediction
            input_data = pd.DataFrame(
                [[weight, height, age, goal_encoded, bmi]],
                columns=['weight', 'height', 'age', 'goal', 'BMI']
            )

            prediction = model.predict(input_data)
            plan = le_plan.inverse_transform(prediction)[0]

            # Rule-based suggestions
            workout = get_workout_plan(category, goal)
            diet = get_diet_plan(category, goal)
            tip = get_tip(goal)   # ✅ ADDED

            result = {
                "bmi": bmi,
                "category": category,
                "plan": plan,
                "workout": workout,
                "diet": diet,
                "tip": tip   # ✅ ADDED
            }

        except Exception as e:
            result = {
                "bmi": "Error",
                "category": "Invalid input",
                "plan": str(e),
                "workout": "-",
                "diet": "-",
                "tip": "-"
            }

    return render_template("index.html", result=result)

# IMPORTANT for Vercel
app.debug = False