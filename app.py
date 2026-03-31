from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
data = pd.read_csv("fitness_data.csv")

# Encoding
le_goal = LabelEncoder()
le_category = LabelEncoder()
le_plan = LabelEncoder()

data['goal'] = le_goal.fit_transform(data['goal'])
data['category'] = le_category.fit_transform(data['category'])
data['plan'] = le_plan.fit_transform(data['plan'])

X = data[['weight', 'height', 'age', 'goal', 'BMI']]
y = data['plan']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Functions
def calculate_bmi(weight, height):
    return round(weight / ((height/100) ** 2), 2)

def get_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    else:
        return "Overweight"

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        age = int(request.form["age"])
        goal = request.form["goal"]

        goal_encoded = le_goal.transform([goal])[0]

        bmi = calculate_bmi(weight, height)
        category = get_category(bmi)

        input_data = pd.DataFrame([[weight, height, age, goal_encoded, bmi]],
                                  columns=['weight', 'height', 'age', 'goal', 'BMI'])

        prediction = model.predict(input_data)
        plan = le_plan.inverse_transform(prediction)[0]

        result = {
            "bmi": bmi,
            "category": category,
            "plan": plan
        }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)