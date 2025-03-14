#!/usr/bin/env python3
"""
Analyse audio en temps réel et affichage du Tonnetz acoustique orthogonal.
Dépendances : pip install numpy scipy matplotlib pyaudio
"""

import threading
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import colorsys
import pyaudio
from scipy.signal import spectrogram, find_peaks

# Configuration et constantes
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

BUFFER_SIZE = 1024         # Échantillons par bloc audio
SAMPLE_RATE = 22050        # Fréquence d'échantillonnage (Hz)
AMPLITUDE_THRESHOLD = 2.0  # Seuil d'amplitude pour détecter un pic

# Facteur de normalisation pour l'affichage des amplitudes
# Ce facteur est choisi selon un empirisme (10 fois le seuil) pour calibrer la visualisation.
NORMALIZATION_FACTOR = AMPLITUDE_THRESHOLD * 10  

GRID_WIDTH = 12            # Largeur du Tonnetz (nombre de colonnes dans la grille)
GRID_HEIGHT = 8            # Hauteur du Tonnetz (nombre de lignes dans la grille)
HUE_SATURATION = 0.5       # Saturation pour la palette de couleurs
HUE_VALUE = 0.5            # Luminosité pour la palette de couleurs

UPDATE_INTERVAL_SEC = 0.01 # Intervalle de rafraîchissement de l'affichage (secondes)


def compute_pitch_class(i: int, j: int) -> int:
    """
    Calcule la classe de hauteur pour un point (i, j) dans le Tonnetz.
    Le calcul se base sur une combinaison linéaire (ici 7*i + 4*j) modulo 12.
    """
    return (7 * i + 4 * j) % 12

def note_to_label(pitch_class: int) -> str:
    """
    Convertit une classe de hauteur (0 à 11) en notation hexadécimale.
    Par exemple, 10 devient 'A' (ou 'A' en majuscule).
    """
    return hex(pitch_class)[2:].upper()

def build_tonnetz_grid() -> tuple:
    """
    Construit la grille du Tonnetz et les bornes pour l'affichage.
    
    Retourne un tuple contenant:
      - tonnetz_grid  : tableau 2D des classes de hauteur,
      - x_boundaries  : bornes pour l'axe horizontal,
      - y_boundaries  : bornes pour l'axe vertical.
    """
    # Plages centrées autour de 0 (pour une meilleure symétrie)
    range_i = np.arange(-GRID_WIDTH // 2, GRID_WIDTH // 2)
    range_j = np.arange(-GRID_HEIGHT // 2, GRID_HEIGHT // 2)
    tonnetz_grid = np.array([[compute_pitch_class(i, j) for i in range_i]
                             for j in range_j])
    x_boundaries = np.arange(range_i.min() - 0.5, range_i.max() + 1, 1)
    y_boundaries = np.arange(range_j.min() - 0.5, range_j.max() + 1, 1)
    return tonnetz_grid, x_boundaries, y_boundaries

def get_rainbow_colormap() -> mcolors.ListedColormap:
    """
    Crée et retourne une colormap en arc-en-ciel pour 12 classes.
    Chaque note reçoit une couleur différente selon la roue HSV.
    """
    colors = [colorsys.hsv_to_rgb(i / 12, HUE_SATURATION, HUE_VALUE)
              for i in range(12)]
    return mcolors.ListedColormap(colors)

def get_piano_colormap() -> mcolors.ListedColormap:
    """
    Crée une colormap inspirée du clavier piano.
    Utilise une palette simple avec 'lightgray' pour certaines notes et 'darkgray' pour d'autres.
    La correspondance est définie en fonction d'une gamme diatonique simplifiée.
    """
    mapping = {(5 + n * 7) % 12: 'lightgray' if n <= 6 else 'darkgray'
               for n in range(12)}
    colors = [mapping[i] for i in range(12)]
    return mcolors.ListedColormap(colors)

def frequency_to_midi(freq: float) -> float:
    """
    Convertit une fréquence (en Hz) en note MIDI.
    La référence standard est 440 Hz pour le A4, correspondant à MIDI 69.
    
    Si la fréquence fournie est <= 0, la fonction retourne 0.0 (valeur par défaut).
    """
    if freq <= 0:
        return 0.0
    return 69 + 12 * np.log2(freq / 440.0)

class AudioStream:
    """
    Gestion du flux audio via PyAudio.
    """
    def __init__(self):
        self.p = pyaudio.PyAudio()
        try:
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=SAMPLE_RATE,
                                      input=True,
                                      frames_per_buffer=BUFFER_SIZE)
        except Exception as e:
            logging.error("Erreur à l'ouverture du flux audio : %s", e)
            raise

    def read(self) -> bytes:
        """
        Lit un bloc d'échantillons audio depuis le flux.
        Retourne les données sous forme de bytes.
        """
        try:
            return self.stream.read(BUFFER_SIZE, exception_on_overflow=False)
        except Exception as e:
            logging.error("Erreur lors de la lecture audio : %s", e)
            return b''

    def close(self):
        """
        Ferme proprement le flux audio.
        """
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

def process_chunk(data: bytes):
    """
    Traite un bloc audio :
      - Conversion des données brutes en tableau NumPy (float32)
      - Calcul du spectrogramme à l'aide d'une fenêtre de Hann (sans chevauchement)
      - Détection des pics d'amplitude qui dépassent AMPLITUDE_THRESHOLD
      - Conversion des pics en tuples (note_midi, amplitude)
      
    Retourne une liste de tuples (note_midi, amplitude) ou None si aucun pic n'est détecté.
    """
    if not data or len(data) < 2:
        return None

    signal = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    if signal.size == 0:
        return None

    # Calcul d'un spectrogramme sur une seule fenêtre complète
    frequencies, _, magnitude = spectrogram(signal,
                                              fs=SAMPLE_RATE,
                                              window='hann',
                                              nperseg=len(signal),
                                              noverlap=0,
                                              mode='magnitude')
    spectrum = magnitude[:, 0]
    peaks, properties = find_peaks(spectrum, height=AMPLITUDE_THRESHOLD)
    if peaks.size == 0:
        return None

    notes = []
    for peak in peaks:
        freq = frequencies[peak]
        # Correspondance du pic à son amplitude (le premier élément suffit)
        amp = properties['peak_heights'][np.where(peaks == peak)][0]
        note = frequency_to_midi(freq)
        notes.append((note, amp))
    return notes

class TonnetzDisplay:
    """
    Affichage interactif du Tonnetz acoustique.
    Gère l'affichage de la grille et la superposition des notes détectées.
    """
    def __init__(self):
        self.tonnetz_grid, self.x_boundaries, self.y_boundaries = build_tonnetz_grid()
        # Tu peux choisir get_rainbow_colormap() ou get_piano_colormap()
        self.colormap = get_piano_colormap()
        self._notes_lock = threading.Lock()  # Verrou pour la gestion thread-safe de detected_notes.
        self.detected_notes = []  # Liste des tuples (note_midi, amplitude)
        self.running = True     # Attribut pour gérer proprement l'arrêt de la boucle d'affichage

    def set_notes(self, notes):
        """
        Met à jour la liste des notes détectées de façon thread-safe.
        """
        with self._notes_lock:
            self.detected_notes = notes.copy() if notes else []

    def get_notes(self):
        """
        Récupère de manière thread-safe la liste des notes détectées.
        """
        with self._notes_lock:
            return self.detected_notes.copy()
            
    def update(self, ax):
        """
        Met à jour l'affichage du Tonnetz :
         - Affiche la grille de la structure Tonnetz
         - Affiche des étiquettes pour chaque case (note hexadécimale)
         - Superpose un cercle sur les cases correspondant aux notes détectées,
           avec une position ajustée par la déviation (offset) calculée à partir des "cents" de différence.
        """
        current_notes = self.get_notes()

        ax.clear()
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("Tonnetz Acoustique Orthogonal", fontsize=16)

        # Affichage de la grille
        ax.pcolormesh(self.x_boundaries, self.y_boundaries,
                      self.tonnetz_grid, cmap=self.colormap,
                      edgecolors='black')

        num_rows, num_cols = self.tonnetz_grid.shape
        # Génération des positions pour les cases (centrées sur des entiers)
        x_positions = np.linspace(-num_cols // 2, num_cols // 2 - 1, num_cols)
        y_positions = np.linspace(-num_rows // 2, num_rows // 2 - 1, num_rows)

        for row_idx, y in enumerate(y_positions):
            for col_idx, x in enumerate(x_positions):
                pitch_class = self.tonnetz_grid[row_idx, col_idx]
                # Affiche le label de chaque case
                ax.text(x, y, note_to_label(pitch_class),
                        ha='center', va='center', color='black', fontsize=12)

                # Pour chaque note détectée, vérifier si elles correspondent à la case (modulo 12)
                for note, amp in current_notes:
                    if round(note) % 12 == pitch_class:
                        # Calcul de la déviation en cents depuis la note arrondie
                        deviation_cents = (note - round(note)) * 100
                        # Définir une amplitude d'offset maximale à appliquer (exemple empirique)
                        offset_max = 0.35
                        # L'offset est proportionnel à la déviation et négatif ici pour un effet de translation
                        offset = - (deviation_cents / 50) * offset_max
                        point_x = x + offset
                        point_y = y + offset
                        # Ajustement de l'alpha pour limiter sa valeur maximale (transparence liée à l'amplitude)
                        alpha = min(0.5, amp / NORMALIZATION_FACTOR)
                        ax.add_patch(patches.Circle((point_x, point_y),
                                                    radius=1/8,
                                                    alpha=alpha,
                                                    color='white'))

    def run(self):
        """
        Boucle d'affichage interactif :
         - Active le mode interactif de Matplotlib, et met à jour la figure.
         - La boucle se termine proprement si l'utilisateur ferme la fenêtre.
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 8))
        try:
            while plt.fignum_exists(fig.number) and self.running:
                self.update(ax)
                plt.draw()
                plt.pause(UPDATE_INTERVAL_SEC)
        except Exception as e:
            logging.error("Erreur d'affichage : %s", e)
        finally:
            plt.ioff()
            plt.show()


def audio_listener(display: TonnetzDisplay):
    """
    Boucle d'écoute audio :
      - Lit continuellement des blocs audio.
      - Traite chaque bloc pour en extraire les notes.
      - Met à jour l'affichage via display.set_notes().
    
    Cette boucle tourne indéfiniment jusqu'à l'arrêt du programme.
    """
    stream = AudioStream()
    try:
        while True:
            data = stream.read()
            notes = process_chunk(data)
            display.set_notes(notes)
    except Exception as e:
        logging.error("Erreur dans le thread audio : %s", e)
    finally:
        stream.close()

def main():
    """
    Orchestration globale du programme :
      1. Initialisation de l'affichage Tonnetz.
      2. Démarrage du thread d'écoute audio.
      3. Lancement de la boucle d'affichage interactif.
      
    Le programme se termine proprement lors d'un KeyboardInterrupt.
    """
    display = TonnetzDisplay()
    audio_thread = threading.Thread(target=audio_listener,
                                    args=(display,),
                                    daemon=True)
    audio_thread.start()
    logging.info("Démarrage de l'analyse audio et de l'affichage.")
    try:
        display.run()
    except KeyboardInterrupt:
        logging.info("Arrêt du programme par l'utilisateur.")
    finally:
        display.running = False

if __name__ == '__main__':
    main()
