
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from gtts import gTTS
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
import numpy as np
import threading
import os

# ---------------- SETTINGS ----------------

model = None
selected_image = None
image_photo = None

# ---------------- LOAD AI MODEL ----------------

def load_model():
    global model
    try:
        model = MobileNetV2(weights="imagenet")
        detector_status.config(
            text="AI model is ready!",
            fg="#22c55e"
        )
    except Exception as e:
        detector_status.config(
            text="Could not load AI model.",
            fg="#ef4444"
        )
        messagebox.showerror("Model Error", str(e))


# ---------------- TEXT TO SPEECH ----------------

def speak_text():
    text = text_input.get("1.0", tk.END).strip()

    if not text:
        messagebox.showwarning(
            "Warning",
            "Please enter some text!"
        )
        return

    speak_button.config(state=tk.DISABLED)
    speech_status.config(
        text="Converting text to speech...",
        fg="#facc15"
    )

    threading.Thread(
        target=convert_to_speech,
        args=(text,),
        daemon=True
    ).start()


def convert_to_speech(text):
    try:
        speech = gTTS(text=text, lang="en")
        speech.save("speech.mp3")

        os.startfile("speech.mp3")

        root.after(
            0,
            lambda: speech_status.config(
                text="Speech completed!",
                fg="#22c55e"
            )
        )

    except Exception as e:
        root.after(
            0,
            lambda: messagebox.showerror("Error", str(e))
        )

    finally:
        root.after(
            0,
            lambda: speak_button.config(state=tk.NORMAL)
        )


# ---------------- CAT AND DOG DETECTOR ----------------

def choose_image():
    global selected_image, image_photo

    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png")
        ]
    )

    if not file_path:
        return

    selected_image = Image.open(file_path).convert("RGB")

    preview = selected_image.copy()
    preview.thumbnail((300, 220))

    image_photo = ImageTk.PhotoImage(preview)
    image_display.config(image=image_photo)

    detector_status.config(
        text="Image loaded. Ready to detect!",
        fg="#60a5fa"
    )

    result_label.config(text="")


def detect_animal():
    if selected_image is None:
        messagebox.showwarning(
            "Warning",
            "Please select an image first!"
        )
        return

    if model is None:
        messagebox.showwarning(
            "Warning",
            "AI model is still loading. Please wait!"
        )
        return

    detect_button.config(state=tk.DISABLED)

    detector_status.config(
        text="AI is analyzing the image...",
        fg="#facc15"
    )

    threading.Thread(
        target=analyze_image,
        daemon=True
    ).start()


def analyze_image():
    try:
        img = selected_image.copy()
        img = img.resize((224, 224))

        image_array = np.array(img)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)

        predictions = model.predict(
            image_array,
            verbose=0
        )

        results = decode_predictions(
            predictions,
            top=5
        )[0]

        animal = None
        confidence = 0

        for _, label, score in results:
            label = label.lower()

            if "cat" in label:
                animal = "CAT"
                confidence = score * 100
                break

            elif "dog" in label:
                animal = "DOG"
                confidence = score * 100
                break

        root.after(
            0,
            lambda: show_result(animal, confidence)
        )

    except Exception as e:
        root.after(
            0,
            lambda: messagebox.showerror(
                "Detection Error",
                str(e)
            )
        )
        root.after(
            0,
            lambda: detect_button.config(state=tk.NORMAL)
        )


def show_result(animal, confidence):
    if animal:
        result_label.config(
            text=f"{animal} detected!\nConfidence: {confidence:.2f}%",
            fg="#22c55e"
        )
    else:
        result_label.config(
            text="This image is not recognized as a cat or dog.",
            fg="#facc15"
        )

    detector_status.config(
        text="Detection completed!",
        fg="#60a5fa"
    )

    detect_button.config(state=tk.NORMAL)


# ---------------- MAIN WINDOW ----------------

root = tk.Tk()
root.title("AI Personal Project")
root.geometry("900x850")
root.configure(bg="#101827")

title = tk.Label(
    root,
    text="AI PERSONAL PROJECT",
    font=("Arial", 25, "bold"),
    bg="#101827",
    fg="#60a5fa"
)
title.pack(pady=20)


# ---------------- TEXT TO SPEECH SECTION ----------------

speech_title = tk.Label(
    root,
    text="TEXT TO SPEECH",
    font=("Arial", 17, "bold"),
    bg="#101827",
    fg="white"
)
speech_title.pack(pady=10)

text_input = tk.Text(
    root,
    width=65,
    height=5,
    font=("Arial", 11),
    bg="#1e293b",
    fg="white",
    insertbackground="white"
)
text_input.pack(pady=10)

speak_button = tk.Button(
    root,
    text="SPEAK",
    command=speak_text,
    bg="#7c3aed",
    fg="white",
    font=("Arial", 11, "bold"),
    width=20
)
speak_button.pack(pady=5)

speech_status = tk.Label(
    root,
    text="Ready!",
    bg="#101827",
    fg="white"
)
speech_status.pack(pady=5)


# ---------------- CAT AND DOG SECTION ----------------

detector_title = tk.Label(
    root,
    text="CAT & DOG DETECTOR",
    font=("Arial", 17, "bold"),
    bg="#101827",
    fg="white"
)
detector_title.pack(pady=15)

image_display = tk.Label(
    root,
    bg="#1e293b",
    width=40,
    height=10
)
image_display.pack(pady=10)

upload_button = tk.Button(
    root,
    text="UPLOAD IMAGE",
    command=choose_image,
    bg="#2563eb",
    fg="white",
    font=("Arial", 11, "bold"),
    width=20
)
upload_button.pack(pady=8)

detect_button = tk.Button(
    root,
    text="DETECT ANIMAL",
    command=detect_animal,
    bg="#7c3aed",
    fg="white",
    font=("Arial", 11, "bold"),
    width=20
)
detect_button.pack(pady=8)

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 15, "bold"),
    bg="#101827",
    fg="#22c55e"
)
result_label.pack(pady=10)

detector_status = tk.Label(
    root,
    text="Loading AI model...",
    bg="#101827",
    fg="white",
    font=("Arial", 10)
)
detector_status.pack(pady=5)


# ---------------- START APPLICATION ----------------

threading.Thread(
    target=load_model,
    daemon=True
).start()

root.mainloop()
