import json
import tkinter as tk
from tkinter import ttk
import numpy as np
import tensorflow as tf

from PIL import Image
from pathlib import Path
from hashlib import md5

MARKER_SIZE = 12
IMAGE_SIZE = 28
CANVAS_SIZE = 280

# CONFIG
JOB_FILE = "./digits.json"
#JOB_FILE = "./tictactoe.json"
MODEL    = tf.keras.models.load_model("model")


def create_img():
    return np.full((CANVAS_SIZE, CANVAS_SIZE), 255, dtype=np.uint8)
    

class App(tk.Frame):
    def __init__(self, parent, job_name=None, labels=None):
        super().__init__()
        self.parent = parent
        self.job_name = job_name
        self.labels = labels
        self.img = create_img()
        self.is_test = tk.IntVar()
        self.pred = tk.IntVar()
        self.is_test.set(0)

        self.build()

    def draw(self, event):
        x, y = event.x, event.y

        def floor(coord):
            for dim in range(2):
                if coord[dim] < 0:
                    coord[dim] = 0
                if coord[dim] > CANVAS_SIZE:
                    coord[dim] = CANVAS_SIZE
            return coord

        tl = [int(x - MARKER_SIZE/2), int(y - MARKER_SIZE/2)]
        br = [int(x + MARKER_SIZE/2), int(y + MARKER_SIZE/2)]
        tl = floor(tl)
        br = floor(br)

        self.fill_rect(tl, br)
        self.img[tl[1]:br[1], tl[0]:br[0]] = 0

    def clear(self):
        self.img = create_img()
        self.canvas.create_rectangle(0, 0, CANVAS_SIZE, CANVAS_SIZE, fill="white")
        self.focus()

    def predict(self):
        im = Image.fromarray(self.img)
        im.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        y_pred = MODEL.predict(np.array(im).reshape(1, 28, 28, 1))
        label = np.argmax(y_pred)
        self.pred.set(label)
        self.focus()

    def save(self, label):
        print(f"INFO: Save to image as '{label}'")
        if self.is_test.get():
            path = Path(f"./{self.job_name}/test/{label}")
        else:
            path = Path(f"./{self.job_name}/train/{label}")

        path.mkdir(parents=True, exist_ok=True)
        filename = md5(self.img.tobytes()).hexdigest()
        filename = f"{filename}.png"
        path = path / filename
        im = Image.fromarray(self.img)
        im.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        im.save(path)
        im.save(f"last_{self.job_name}.png")
        self.clear()
        self.update()

    def fill_rect(self, tl, br):
        self.canvas.create_rectangle(
            tl[0], tl[1],
            br[0], br[1],
            fill="black"
        )

    def btn_clicked(self, event):
        label = event.widget.cget('text')
        self.focus()
        self.save(label)

    def key_pressed(self, event):
        key = event.keysym
        for label in self.labels:
            if label["key"] == key:
                self.save(key)

    def build(self):
        self.bind("<Key>", self.key_pressed)
        self.focus()

        tframe = tk.Frame(self)
        lframe = tk.Frame(self)
        mframe = tk.Frame(self)
        rframe = tk.Frame(self)

        job_label = ttk.Label(tframe, text=f"Job: {self.job_name}")
        job_label.pack()

        self.canvas = tk.Canvas(mframe, width=CANVAS_SIZE, height=CANVAS_SIZE)
        self.clear()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.pack()

        predict_btn = ttk.Button(
            lframe,
            text="►",
            command=self.predict
        )
        pred_label = ttk.Label(lframe, textvariable=self.pred)
        clear_btn = ttk.Button(lframe, text="🗑️", command=self.clear)
        test_check_btn = ttk.Checkbutton(lframe, text="Test-Set", variable=self.is_test, command=self.update)

        predict_btn.pack()
        pred_label.pack()
        clear_btn.pack()
        test_check_btn.pack()

        self.amounts_per_class = [tk.StringVar() for _ in range(10)]
        for i, label in enumerate(self.labels):
            wrapper = tk.Frame(rframe)
            btn = ttk.Button(wrapper, text=label["name"], command=self.update)
            btn.bind("<Button-1>", self.btn_clicked)
            btn.pack(side="left")
            tk.Label(wrapper, textvariable=self.amounts_per_class[i]).pack(side="left")
            wrapper.pack()

        tframe.pack()
        lframe.pack(side="left", fill="y")
        mframe.pack(side="left", fill="y")
        rframe.pack(side="left", fill="y")
        self.pack(padx=10, pady=10)

        self.update()

    def update(self):
        for i, label in enumerate(self.labels):
            data_path_train = Path(f"./{self.job_name}/train/{label['name']}")
            data_path_test = Path(f"./{self.job_name}/test/{label['name']}")
            amount_train = len(list(data_path_train.glob("*.png")))
            amount_test = len(list(data_path_test.glob("*.png")))
            amount = f"{amount_train} | {amount_test}"
            self.amounts_per_class[i].set(amount)
        self.focus()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Symbol Labeling Tool")

    with open(JOB_FILE) as f:
        job = json.load(f)

    app = App(
        root,
        job_name=job["name"],
        labels=job["labels"]
    )

    root.mainloop()
