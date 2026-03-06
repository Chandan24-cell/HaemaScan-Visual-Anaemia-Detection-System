from supabase import create_client
from flask import Flask, render_template, request, redirect, session, url_for, jsonify
import numpy as np
import joblib
from process_image import process_image
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load pre-trained model
print("Loading model...")
model = joblib.load('model/random_forest_classifier.pkl')
print("Model loaded successfully.")


@app.route('/login/email-password', methods=['GET', 'POST'])
def login_email_password():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
 
        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            session['user'] = response.user.email
            return redirect(url_for('dashboard'))

        except Exception as e:
            return render_template('login.html', error='Invalid email or password')

    return render_template('login.html')


@app.route('/login/google')
def login_google():
    try:
        res = supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": url_for("auth_callback", _external=True)
            }
        })
        return redirect(res.url)
    except Exception as e:
        return render_template('login.html', error="Google login failed")


@app.route('/login', methods=['GET', 'POST'])
def login():
    # Redirect to email-password login page by default
    return redirect(url_for('login_email_password'))


@app.route('/auth/callback')
def auth_callback():
    # Mark user as logged in
    session["logged_in"] = True

    # (Optional) store email if you want later
    session["user"] = "google_user"

    return redirect(url_for("dashboard"))


@app.route('/', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    result = None
    if request.method == 'POST':
        print("POST request received.")
        try:
            # Debugging: Check form values
            print("Form data:", request.form)

            hemoglobin = float(request.form['hemoglobin'])
            mcv = float(request.form['mcv'])
            mch = float(request.form['mch'])
            mchc = float(request.form['mchc'])
            gender = request.form['gender']

            # Convert gender to binary (0 for Female, 1 for Male)
            gender_binary = 1 if gender.lower() == 'male' else 0
            print(f"Parsed inputs: Hemoglobin={hemoglobin}, MCV={mcv}, MCH={mch}, MCHC={mchc}, Gender={gender} (Binary={gender_binary})")

            # Prepare the input features
            features = np.array([[gender_binary, hemoglobin, mch, mchc, mcv]])
            print("Prepared features:", features)

            # Predict the result
            prediction = model.predict(features)[0]
            print("Model prediction:", prediction)

            result = 'Anemic' if prediction == 1 else 'Not Anemic'
            print("Result:", result)

        except Exception as e:
            print("Error occurred:", str(e))
            result = f"Error: {e}"

    # Debugging: Check final result before rendering
        print("Final result to render:", result)
        return jsonify({"result": result})
    return render_template('index.html', result=result)


@app.route("/logout")
def logout():
    session.clear()   # clears all session data
    return redirect(url_for("login"))


@app.route('/predict-image', methods=['POST'])
def upload_image():
    try:
        print("Endpoint '/predict-image' hit.")  # Debug: Endpoint reached
        
        # Check if an image was uploaded
        if 'image' not in request.files:
            print("Debug: No image key in request files.")  # Debug: Missing 'image'
            return jsonify({"error": "No image uploaded"}), 400

        uploaded_file = request.files['image']
        print(f"Debug: Uploaded file name - {uploaded_file.filename}")  # Debug: File name
        
        if uploaded_file.filename == '':
            print("Debug: Uploaded file name is empty.")  # Debug: Empty file name
            return jsonify({"error": "No selected file"}), 400

        # Read the uploaded image
        image_bytes = uploaded_file.read()
        print("Debug: Image bytes read successfully.")  # Debug: Image read

        # Pass the image bytes to `process_image` function
        extracted_features = process_image(image_bytes)
        print("Debug: Extracted features:", extracted_features)  # Debug: Features extracted

        # Extract features from the dictionary
        gender_binary = extracted_features.get('gender')
        hemoglobin = extracted_features.get('hemoglobin')
        mch = extracted_features.get('mch')
        mchc = extracted_features.get('mchc')
        mcv = extracted_features.get('mcv')

        print(f"Debug: Extracted values - Gender: {gender_binary}, Hemoglobin: {hemoglobin}, MCH: {mch}, MCHC: {mchc}, MCV: {mcv}")  # Debug: Feature values

        # Check for missing or invalid features
        if None in [gender_binary, hemoglobin, mch, mchc, mcv]:
            print("Debug: One or more extracted features are missing or invalid.")  # Debug: Missing features
            return jsonify({"error": "Invalid features extracted from image"}), 400

        # Prepare the input features for the model
        features = np.array([[gender_binary, hemoglobin, mch, mchc, mcv]])
        print("Debug: Features prepared for model:", features)  # Debug: Features for model

        # Get prediction from the model
        prediction = model.predict(features)[0]
        result = 'Anemic' if prediction == 1 else 'Not Anemic'
        print("Debug: Prediction result:", result)  # Debug: Prediction result

        # Return the prediction result as JSON
        response = jsonify({"result": result, "features": extracted_features})
        print("Debug: Response prepared successfully.")  # Debug: Response ready
        return response

    except Exception as e:
        print("Error during image processing:", str(e))  # Debug: Exception details
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))  # 5001 for local, Render overrides it
    app.run(host="0.0.0.0", port=port, debug=True)
