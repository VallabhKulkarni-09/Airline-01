import pickle
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
model = pickle.load(open('classifier.pkl', 'rb'))

# Helper function to preprocess user inputs
def preprocess_inputs(inputs):
    # Map user inputs to the expected format
    gender_mapping = {'Male': 1, 'Female': 0}
    customer_type_mapping = {'Loyal Customer': 0, 'Disloyal Customer': 1}
    travel_type_mapping = {'Personal Travel': 1, 'Business travel': 0}
    class_mapping = {'Business': 0, 'Eco': 1, 'Eco Plus': 2}

    inputs['Gender'] = gender_mapping.get(inputs['Gender'], 0)
    inputs['Customer Type'] = customer_type_mapping.get(inputs['Customer Type'], 0)
    inputs['Type of Travel'] = travel_type_mapping.get(inputs['Type of Travel'], 0)
    inputs['Class'] = class_mapping.get(inputs['Class'], 0)

    return inputs

# Helper function to predict satisfaction level
def predict_satisfaction(inputs):
    inputs = np.array(list(inputs.values())).reshape(1, -1)
    prediction = model.predict(inputs)
    return prediction[0]

# Helper function to interpret the prediction
def interpret_prediction(prediction):
    if prediction == 0:
        return 'Unsatisfied or Neutral'
    elif prediction == 1:
        return 'Satisfied'
    else:
        return 'Unknown'

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_inputs = {
            'Gender': request.form['gender'],
            'Customer Type': request.form['customer_type'],
            'Age': int(request.form['age']),
            'Type of Travel': request.form['type_of_travel'],
            'Class': request.form['class'],
            'Flight Distance': int(request.form['flight_distance']),
            'Departure Delay in Minutes': int(request.form['departure_delay']),
            'Arrival Delay in Minutes': int(request.form['arrival_delay']),
            'Inflight wifi service': int(request.form['wifi_service']),
            'Departure/Arrival time convenient': int(request.form['time_convenience']),
            'Ease of Online booking': int(request.form['online_booking']),
            'Gate location': int(request.form['gate_location']),
            'Food and drink': int(request.form['food_and_drink']),
            'Online boarding': int(request.form['online_boarding']),
            'Seat comfort': int(request.form['seat_comfort']),
            'Inflight entertainment': int(request.form['entertainment']),
            'On-board service': int(request.form['on_board_service']),
            'Leg room service': int(request.form['leg_room_service']),
            'Baggage handling': int(request.form['baggage_handling']),
            'Checkin service': int(request.form['checkin_service']),
            'Inflight service': int(request.form['inflight_service']),
            'Cleanliness': int(request.form['cleanliness'])
        }

        # Preprocess inputs
        user_inputs = preprocess_inputs(user_inputs)

        # Predict satisfaction level
        prediction = predict_satisfaction(user_inputs)

        # Interpret prediction
        result = interpret_prediction(prediction)

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
