import matplotlib.pyplot as plt
import torch
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageDraw, ImageOps
import numpy as np

from python import data


# -----------------------------
# GUI Setup
# -----------------------------
class DrawingApp:
    def __init__(self, master, model, kernel):
        self.model = model
        self.diffuser = data.DiffuserTransform(kernel)

        self.master = master
        self.master.title("Draw a digit")
        self.canvas_size = 500
        self.canvas = tk.Canvas(master, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()

        # Button to predict
        self.button = tk.Button(master, text="Predict", command=self.predict)
        self.button.pack()

        # Button to clear
        self.clear_button = tk.Button(master, text="Clear", command=self.clear)
        self.clear_button.pack()

        # PIL image to draw on
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)

        # Mouse events
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 20), (event.y - 20)
        x2, y2 = (event.x + 20), (event.y + 20)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=255)

    def preprocess_image(self):
        # Resize to 28x28, invert colors, normalize
        img = self.image.resize((16, 16), Image.Resampling.LANCZOS)
        img = ImageOps.invert(img)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
        img = self.diffuser(img)
        return img

    def predict(self):
        img_tensor = self.preprocess_image()
        with torch.no_grad():
            output = self.model(img_tensor)
            plt.figure()
            plt.imshow(output.squeeze(0).squeeze(0), cmap="grey")
            plt.show()
