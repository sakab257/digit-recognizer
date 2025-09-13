from keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2

# Chargement du modèle
try:
    model = load_model('mnist_improved.h5')
    print("Modèle amélioré chargé")
except:
    exit

def preprocess_image(img):
    """
    Prétraitement amélioré pour correspondre au format MNIST
    """
    # Conversion en niveaux de gris
    img = img.convert('L')
    
    # Conversion en array numpy
    img_array = np.array(img)
    
    # IMPORTANT: Inversion des couleurs (MNIST = blanc sur noir)
    img_array = 255 - img_array
    
    # Trouver la boîte englobante du dessin
    coords = cv2.findNonZero(img_array)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        
        # Extraire la région d'intérêt avec un peu de padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_array.shape[1] - x, w + 2 * padding)
        h = min(img_array.shape[0] - y, h + 2 * padding)
        
        roi = img_array[y:y+h, x:x+w]
        
        # Redimensionner en gardant le ratio d'aspect
        if h > w:
            new_h = 20
            new_w = int(w * (20 / h))
        else:
            new_w = 20
            new_h = int(h * (20 / w))
        
        if new_w > 0 and new_h > 0:
            roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Créer une image 28x28 avec le chiffre centré
            img_array = np.zeros((28, 28), dtype=np.uint8)
            y_offset = (28 - new_h) // 2
            x_offset = (28 - new_w) // 2
            img_array[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi
        else:
            # Si les dimensions sont invalides, redimensionner directement
            img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        # Si aucun dessin n'est détecté, redimensionner directement
        img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalisation
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape pour le modèle
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

def predict_digit(img):
    """
    Prédiction améliorée avec preprocessing approprié
    """
    # Prétraitement de l'image
    processed_img = preprocess_image(img)
    
    # Prédiction
    predictions = model.predict(processed_img, verbose=0)[0]
    
    # Obtenir les top 3 prédictions
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    top_3_prob = predictions[top_3_idx]
    
    return top_3_idx[0], predictions[top_3_idx[0]], list(zip(top_3_idx, top_3_prob))

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        
        self.title("Reconnaissance de Chiffres IA")
        self.resizable(False, False)
        self.configure(bg="#f8f9fa")
        
        # Variables pour le dessin
        self.old_x = None
        self.old_y = None
        self.line_width = 18
        
        # Frame principal avec padding
        main_frame = tk.Frame(self, bg="#f8f9fa", padx=30, pady=25)
        main_frame.grid(row=0, column=0)
        
        # Titre principal
        title_label = tk.Label(main_frame, text="Reconnaissance de Chiffres", 
                              font=("Segoe UI", 24, "bold"), fg="#2c3e50", bg="#f8f9fa")
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Canvas avec bordure moderne
        canvas_frame = tk.Frame(main_frame, bg="#ffffff", relief="solid", bd=1)
        canvas_frame.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        self.canvas = tk.Canvas(canvas_frame, width=340, height=340, bg="#ffffff", 
                               cursor="crosshair", highlightthickness=2, highlightcolor="#007acc", bd=0)
        self.canvas.grid(row=0, column=0, padx=8, pady=8)
        
        # Zone des résultats avec style card moderne
        results_frame = tk.Frame(main_frame, bg="#ffffff", relief="solid", bd=1, padx=20, pady=18)
        results_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        results_frame.grid_columnconfigure(0, weight=1)
        
        self.label = tk.Label(results_frame, text="Dessinez un chiffre de 0 à 9", 
                             font=("Segoe UI", 16, "normal"), fg="#6c757d", bg="#ffffff")
        self.label.grid(row=0, column=0, pady=(0, 10))
        
        self.confidence_label = tk.Label(results_frame, text="", 
                                        font=("Segoe UI", 14), fg="#495057", bg="#ffffff")
        self.confidence_label.grid(row=1, column=0, pady=3)
        
        self.alternatives_label = tk.Label(results_frame, text="", 
                                          font=("Segoe UI", 12), fg="#6c757d", bg="#ffffff")
        self.alternatives_label.grid(row=2, column=0, pady=3)
        
        # Boutons avec style moderne
        button_frame = tk.Frame(main_frame, bg="#f8f9fa")
        button_frame.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        self.classify_btn = tk.Button(button_frame, text="Analyser", 
                                     command=self.classify_handwriting,
                                     bg="#007acc", fg="green", font=("Segoe UI", 13, "bold"),
                                     relief="flat", bd=0, padx=30, pady=12,
                                     cursor="hand2",
                                     highlightthickness=0)
        self.classify_btn.pack(side=LEFT, padx=(0, 15))
        
        self.button_clear = tk.Button(button_frame, text="Effacer", 
                                     command=self.clear_all,
                                     bg="#6c757d", fg="red", font=("Segoe UI", 13, "bold"),
                                     relief="flat", bd=0, padx=30, pady=12,
                                     cursor="hand2",
                                     highlightthickness=0)
        self.button_clear.pack(side=LEFT)
        
        # Bindings pour le dessin
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        # Instructions avec style amélioré
        instructions = tk.Label(main_frame, text="Conseil: Dessinez le chiffre bien centré et assez grand pour une meilleure précision", 
                               font=("Segoe UI", 11), fg="#6c757d", bg="#f8f9fa",
                               wraplength=450, justify="center")
        instructions.grid(row=4, column=0, columnspan=2)

    def paint(self, event):
        """Dessiner avec des lignes continues pour un meilleur rendu"""
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                   width=self.line_width, fill='black',
                                   capstyle=tk.ROUND, smooth=tk.TRUE)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        """Réinitialiser les coordonnées"""
        self.old_x = None
        self.old_y = None

    def clear_all(self):
        """Effacer le canvas et réinitialiser les labels"""
        self.canvas.delete("all")
        self.label.configure(text="Dessinez un chiffre de 0 à 9")
        self.confidence_label.configure(text="")
        self.alternatives_label.configure(text="")
        
    def classify_handwriting(self):
        """Classification avec affichage amélioré des résultats"""
        # Créer une image directement à partir du canvas sans PostScript
        im = Image.new('RGB', (300, 300), 'white')
        draw = ImageDraw.Draw(im)
        
        # Récupérer tous les éléments du canvas et les redessiner sur l'image PIL
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) >= 4:  # Ligne
                draw.line(coords, fill='black', width=self.line_width)
        
        # Prédiction
        digit, confidence, top_3 = predict_digit(im)
        
        # Affichage du résultat principal
        if confidence > 0.8:
            conf_color = "#28a745"
            conf_text = "Très confiant"
        elif confidence > 0.5:
            conf_color = "#ffc107"
            conf_text = "Moyennement confiant"
        else:
            conf_color = "#dc3545"
            conf_text = "Peu confiant"
        
        self.label.configure(text=f"Prédiction: {digit}", fg=conf_color)
        self.confidence_label.configure(
            text=f"Confiance: {confidence*100:.1f}% ({conf_text})"
        )
        
        # Affichage des alternatives
        alt_text = "Alternatives: "
        for idx, (d, p) in enumerate(top_3[1:3]):
            alt_text += f"{d} ({p*100:.1f}%)"
            if idx == 0:
                alt_text += ", "
        self.alternatives_label.configure(text=alt_text)
        
        # Optionnel: Sauvegarder l'image prétraitée pour debug
        # processed = preprocess_image(im)
        # debug_img = (processed[0, :, :, 0] * 255).astype(np.uint8)
        # Image.fromarray(debug_img).save('debug_preprocessed.png')

# Fonction utilitaire pour visualiser le preprocessing (debug)
def visualize_preprocessing(canvas_img):
    """Fonction de debug pour visualiser le preprocessing"""
    import matplotlib.pyplot as plt
    
    # Image originale
    plt.figure(figsize=(12, 3))
    
    plt.subplot(1, 4, 1)
    plt.imshow(canvas_img)
    plt.title("Original")
    plt.axis('off')
    
    # Niveaux de gris
    gray = canvas_img.convert('L')
    plt.subplot(1, 4, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.axis('off')
    
    # Inversé
    inverted = 255 - np.array(gray)
    plt.subplot(1, 4, 3)
    plt.imshow(inverted, cmap='gray')
    plt.title("Inverted (MNIST style)")
    plt.axis('off')
    
    # Final preprocessed
    processed = preprocess_image(canvas_img)
    plt.subplot(1, 4, 4)
    plt.imshow(processed[0, :, :, 0], cmap='gray')
    plt.title("Final (28x28)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    app = App()
    app.mainloop()