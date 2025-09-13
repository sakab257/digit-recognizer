from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image, ImageOps
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
        
        self.title("Reconnaissance de chiffres manuscrits")
        self.resizable(False, False)
        
        # Variables pour le dessin
        self.old_x = None
        self.old_y = None
        self.line_width = 15
        
        # Création des éléments
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Dessinez un chiffre...", font=("Helvetica", 24))
        self.confidence_label = tk.Label(self, text="", font=("Helvetica", 12))
        self.alternatives_label = tk.Label(self, text="", font=("Helvetica", 10), fg="gray")
        
        # Boutons
        button_frame = tk.Frame(self)
        self.classify_btn = tk.Button(button_frame, text="Prédire", 
                                     command=self.classify_handwriting,
                                     bg="#4CAF50", fg="white", font=("Helvetica", 12))
        self.button_clear = tk.Button(button_frame, text="Effacer", 
                                     command=self.clear_all,
                                     bg="#f44336", fg="white", font=("Helvetica", 12))
        
        # Disposition
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.label.grid(row=1, column=0, columnspan=2, pady=5)
        self.confidence_label.grid(row=2, column=0, columnspan=2, pady=2)
        self.alternatives_label.grid(row=3, column=0, columnspan=2, pady=2)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        self.classify_btn.pack(side=LEFT, padx=5)
        self.button_clear.pack(side=LEFT, padx=5)
        
        # Bindings pour le dessin
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        # Instructions
        instructions = tk.Label(self, text="Astuce: Dessinez le chiffre au centre et assez grand", 
                               font=("Helvetica", 9), fg="gray")
        instructions.grid(row=5, column=0, columnspan=2, pady=5)

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
        self.label.configure(text="Dessinez un chiffre...")
        self.confidence_label.configure(text="")
        self.alternatives_label.configure(text="")
        
    def classify_handwriting(self):
        """Classification avec affichage amélioré des résultats"""
        # Capture de l'image du canvas
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        x, y, x2, y2 = rect
        
        # Ajustement pour éviter les bordures
        im = ImageGrab.grab((x+2, y+2, x2-2, y2-2))
        
        # Prédiction
        digit, confidence, top_3 = predict_digit(im)
        
        # Affichage du résultat principal
        if confidence > 0.8:
            conf_color = "green"
            conf_text = "Très confiant"
        elif confidence > 0.5:
            conf_color = "orange"
            conf_text = "Moyennement confiant"
        else:
            conf_color = "red"
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