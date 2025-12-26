import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model(r"D:\HandWritten Recoginiton  Project\Project\mnist_cnn_model.h5")

# GUI setup
root = tk.Tk()
root.title("MNIST Digit Recognition (CNN)")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

result_label = tk.Label(frame, text="Draw a digit and click Predict",
                        font=("Arial", 16))
result_label.pack(pady=10)

canvas_size = 280  # user drawing canvas size
canvas = tk.Canvas(frame, width=canvas_size, height=canvas_size,
                   bg="white", cursor="cross")
canvas.pack()

# PIL image to store drawings
image1 = Image.new("L", (canvas_size, canvas_size), "white")
draw = ImageDraw.Draw(image1)

def paint(event):
    x1, y1 = (event.x - 8), (event.y - 8)
    x2, y2 = (event.x + 8), (event.y + 8)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
    draw.ellipse([x1, y1, x2, y2], fill="black")

canvas.bind("<B1-Motion>", paint)

def predict_digit():
    # Convert drawing to 28x28 grayscale
    img = image1.copy()
    img = ImageOps.invert(img)       # Invert (white bg → black digit)
    img = img.resize((28, 28))       # Resize to MNIST size
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0
    pred = model.predict(img)
    digit = np.argmax(pred)
    confidence = np.max(pred) * 100
    result_label.config(text=f"Prediction: {digit}  (Confidence: {confidence:.2f}%)")

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, canvas_size, canvas_size], fill="white")
    result_label.config(text="Draw a digit and click Predict")

btn_frame = tk.Frame(frame)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Predict", command=predict_digit).pack(side=tk.LEFT, padx=5)
tk.Button(btn_frame, text="Clear", command=clear_canvas).pack(side=tk.LEFT, padx=5)

root.mainloop()
