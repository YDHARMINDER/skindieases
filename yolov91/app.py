from flask import Flask, request, render_template
import os
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
model = YOLO("C:/Users/ydhar/runs/detect/yolov8n_custom/weights/best.pt")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    img = Image.open(file.stream)
    imgsz = int(request.form.get('imgsz', 1280))
    show = request.form.get('show', 'on') == 'on'
    hide_labels = request.form.get('hide_labels', 'on') == 'on'

    results = model.predict(source=img, imgsz=imgsz, show=show, hide_labels=hide_labels)
    
    for r in results:
        im_array = r.plot()
        output_img = Image.fromarray(im_array[..., ::-1])
        output_path = os.path.join('static', 'output.jpg')
        output_img.save(output_path)

    return render_template('result.html', output_image='output.jpg')

if __name__ == "__main__":
    app.run(debug=True)