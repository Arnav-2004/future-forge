from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from PIL import Image

app = Flask(__name__)

model = load_model("model.hdf5")

def predict_class(file):
    img = Image.open(file)
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    out = model.predict(img)
    out = np.argmax(out)
    return out

classes = {
    0: 'Apple: Scab', 1: 'Apple: Black Rot', 2: 'Apple: Cedar Apple Rust', 3: 'Apple: Healthy',
    4: 'Blueberry: Healthy', 5: 'Cherry (including sour): Powdery Mildew', 6: 'Cherry (including sour): Healthy',
    7: 'Corn (maize): Cercospora Leaf Spot (Gray Leaf Spot)', 8: 'Corn (maize): Common Rust', 9: 'Corn (maize): Northern Leaf Blight',
    10: 'Corn (maize): Healthy', 11: 'Grape: Black Rot', 12: 'Grape: Esca (Black Measles)',
    13: 'Grape: Leaf Blight (Isariopsis Leaf Spot)', 14: 'Grape: Healthy', 15: 'Orange: Haunglongbing (Citrus Greening)',
    16: 'Peach: Bacterial Spot', 17: 'Peach: Healthy', 18: 'Pepper Bell: Bacterial Spot', 19: 'Pepper Bell: Healthy',
    20: 'Potato: Early Blight', 21: 'Potato: Late Blight', 22: 'Potato: Healthy', 23: 'Raspberry: Healthy',
    24: 'Soybean: Healthy', 25: 'Squash: Powdery Mildew', 26: 'Strawberry: Leaf Scorch', 27: 'Strawberry: Healthy',
    28: 'Tomato: Bacterial Spot', 29: 'Tomato: Early Blight', 30: 'Tomato: Late Blight', 31: 'Tomato: Leaf Mold',
    32: 'Tomato: Septoria Leaf Spot', 33: 'Tomato: Spider Mites (Two-spotted Spider Mite)', 34: 'Tomato: Target Spot',
    35: 'Tomato: Yellow Leaf Curl Virus', 36: 'Tomato: Mosaic Virus', 37: 'Tomato: Healthy'
}

disease_solutions = {
    0: "For Apple Scab, apply fungicides containing chlorothalonil or sulfur. Prune and remove infected plant parts.",
    1: "For Apple Black Rot, prune infected branches and use fungicides like thiophanate-methyl or copper hydroxide.",
    2: "For Apple Cedar Apple Rust, remove affected leaves and use fungicides like myclobutanil or mancozeb.",
    3: "For Healthy Apples, maintain proper sanitation practices, prune infected areas, and ensure proper watering and fertilization.",
    4: "For Healthy Blueberries, maintain proper cultural practices including pruning, watering, and fertilization.",
    5: "For Cherry Powdery Mildew, apply fungicides like sulfur or potassium bicarbonate. Prune infected areas.",
    6: "For Healthy Cherries, maintain proper cultural practices including pruning, watering, and fertilization.",
    7: "For Corn Cercospora Leaf Spot, rotate crops, apply fungicides like chlorothalonil, and remove infected residues.",
    8: "For Corn Common Rust, remove infected plants, apply fungicides containing triazole or strobilurin, and plant resistant varieties.",
    9: "For Corn Northern Leaf Blight, apply fungicides containing strobilurin or triazole, and rotate crops.",
    10: "For Healthy Corn, practice crop rotation, tillage, and use resistant varieties.",
    11: "For Grape Black Rot, prune infected parts, apply fungicides like captan or myclobutanil, and practice proper vineyard sanitation.",
    12: "For Grape Esca (Black Measles), prune infected parts, apply fungicides containing cyprodinil or fludioxonil.",
    13: "For Grape Leaf Blight, apply fungicides containing copper hydroxide or sulfur, and remove infected leaves.",
    14: "For Healthy Grapes, maintain proper vineyard management practices including pruning, watering, and fertilization.",
    15: "For Orange Haunglongbing (Citrus Greening), there's no known cure. Remove infected trees and control psyllid populations.",
    16: "For Peach Bacterial Spot, apply copper-based fungicides, prune infected parts, and practice proper sanitation.",
    17: "For Healthy Peaches, maintain proper cultural practices including pruning, watering, and fertilization.",
    18: "For Pepper Bell Bacterial Spot, apply copper-based fungicides, prune infected parts, and use disease-resistant varieties.",
    19: "For Healthy Pepper Bell, maintain proper cultural practices including pruning, watering, and fertilization.",
    20: "For Potato Early Blight, apply fungicides containing chlorothalonil or mancozeb, and practice crop rotation.",
    21: "For Potato Late Blight, apply fungicides containing chlorothalonil or mancozeb, and practice crop rotation.",
    22: "For Healthy Potatoes, practice crop rotation, remove infected plant parts, and apply fungicides preventatively.",
    23: "For Healthy Raspberries, maintain proper cultural practices including pruning, watering, and fertilization.",
    24: "For Healthy Soybeans, maintain proper cultural practices including crop rotation and planting disease-resistant varieties.",
    25: "For Squash Powdery Mildew, apply fungicides like sulfur or potassium bicarbonate, and maintain good airflow around plants.",
    26: "For Strawberry Leaf Scorch, remove infected leaves, apply fungicides containing captan or mancozeb, and maintain proper irrigation.",
    27: "For Healthy Strawberries, maintain proper cultural practices including pruning, watering, and fertilization.",
    28: "For Tomato Bacterial Spot, apply copper-based fungicides, remove infected plant parts, and practice crop rotation.",
    29: "For Tomato Early Blight, apply fungicides containing chlorothalonil or mancozeb, and remove infected leaves.",
    30: "For Tomato Late Blight, apply fungicides containing chlorothalonil or mancozeb, and remove infected leaves.",
    31: "For Tomato Leaf Mold, apply fungicides containing chlorothalonil or mancozeb, and maintain good airflow around plants.",
    32: "For Tomato Septoria Leaf Spot, apply fungicides containing chlorothalonil or mancozeb, and remove infected leaves.",
    33: "For Tomato Spider Mites, apply miticides or insecticidal soaps, and maintain proper humidity levels.",
    34: "For Tomato Target Spot, apply fungicides containing chlorothalonil or mancozeb, and remove infected leaves.",
    35: "For Tomato Yellow Leaf Curl Virus, apply insecticides to control the vector insects, remove infected plants, and maintain proper sanitation.",
    36: "For Tomato Mosaic Virus, there's no cure. Remove infected plants and control aphid populations.",
    37: "For Healthy Tomatoes, maintain proper cultural practices including pruning, watering, and fertilization."
}


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    file = request.files["file"]
    image_path = 'images/' + file.filename 
    file.save(image_path)
    prediction = predict_class(image_path)
    if prediction in disease_solutions:
        solution = disease_solutions[prediction]
    else:
        solution = "Solution not available for this disease."
    return render_template('index.html', prediction=classes[prediction], solution=solution)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
