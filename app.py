from flask import Flask, render_template, redirect, url_for, request, flash, session
import joblib
import numpy as np
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Dummy credentials
USERNAME = 'admin'
PASSWORD = 'password'

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart_disease.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the trained Decision Tree model
model = joblib.load('decision_tree_model.pkl')

# Create a model for storing user inputs
class UserInput(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False)
    cp = db.Column(db.Integer, nullable=False)
    trestbps = db.Column(db.Integer, nullable=False)
    chol = db.Column(db.Integer, nullable=False)
    fbs = db.Column(db.Integer, nullable=False)
    restecg = db.Column(db.Integer, nullable=False)
    thalach = db.Column(db.Integer, nullable=False)
    exang = db.Column(db.Integer, nullable=False)
    oldpeak = db.Column(db.Float, nullable=False)
    slope = db.Column(db.Integer, nullable=False)
    ca = db.Column(db.Integer, nullable=False)
    thal = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)

# Create the database
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    # Redirect to login page
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('heart_disease_form'))
        else:
            flash('Invalid credentials. Please try again.', 'error')

    return render_template('login.html')

@app.route('/logout')
def logout():
    # Remove the user from the session if logged in
    session.pop('logged_in', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/heart_disease_form', methods=['GET'])
def heart_disease_form():
    if 'logged_in' not in session or not session['logged_in']:
        flash('Please login first.', 'error')
        return redirect(url_for('login'))
    
    return render_template('index.html')

@app.route('/view_database', methods=['GET'])
def view_database():
    if 'logged_in' not in session or not session['logged_in']:
        flash('Please login first.', 'error')
        return redirect(url_for('login'))

    # Fetch all records from UserInput table
    user_inputs = UserInput.query.all()
    return render_template('view_database.html', user_inputs=user_inputs)


@app.route('/predict', methods=['POST'])
def predict():
    if 'logged_in' not in session or not session['logged_in']:
        flash('Please login first.', 'error')
        return redirect(url_for('login'))
        
    data = request.form.to_dict()
    try:
        # Extracting the features including the name
        name = data['name']
        age = int(data['age'])
        sex = int(data['sex'])
        cp = int(data['cp'])
        trestbps = int(data['trestbps'])
        chol = int(data['chol'])
        fbs = int(data['fbs'])
        restecg = int(data['restecg'])
        thalach = int(data['thalach'])
        exang = int(data['exang'])
        oldpeak = float(data['oldpeak'])
        slope = int(data['slope'])
        ca = int(data['ca'])
        thal = int(data['thal'])

        # Prepare features for the model
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Use the loaded model for prediction
        prediction = model.predict(features)[0]

       # Interpret the prediction
        if prediction == 0:
             result = 'No Risk'  # No Risk
             tips = "Great! Keep up the healthy lifestyle, and continue regular check-ups."
        else:
             result = 'Risk'  # Risk
             tips = "You are at risk. Consider consulting a healthcare professional and adopting a heart-friendly lifestyle."


        # Save user input to the database
        user_input = UserInput(name=name, age=age, sex=sex, cp=cp, trestbps=trestbps,
                                chol=chol, fbs=fbs, restecg=restecg, thalach=thalach,
                                exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal,
                                prediction=result)

        db.session.add(user_input)
        db.session.commit()

        # Render the result page and pass prediction, age, tips, and name
        return render_template('result.html', name=name, age=age,prediction=result, tips=tips)

    except Exception as e:
        flash(f'Error in prediction: {str(e)}', 'error')
        return redirect(url_for('heart_disease_form'))

if __name__ == '__main__':
    app.run(debug=True)
