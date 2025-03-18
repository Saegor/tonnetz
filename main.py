"""
Ce script met en place une analyse audio en temps réel permettant d’extraire et d’afficher les notes détectées
sur deux représentations : une grille et un cercle.
Il utilise pour cela la transformée de Fourier, la détection de pic et la conversion vers une note MIDI.
- Le flux audio sous forme de bytes est décodé en entiers (par exemple, en int16).
- Les données sont ensuite converties en format float32, pour faciliter les traitements numériques ultérieurs.
- Un filtre (ex. passe-bas Butterworth) est appliqué pour atténuer les composantes indésirables du signal (bruit).
- Le filtrage est réalisé, par exemple, avec filtfilt qui évite le déphasage, garantissant une meilleure fidélité temporelle.
- Le signal est découpé en trames (chunks) de taille fixe (BUF_SIZE) afin d’effectuer une analyse par blocs temporels.
- Un fenêtrage (comme une fenêtre Hann) est appliqué sur chaque segment pour réduire les effets de fuite spectrale et améliorer la qualité de la FFT.
- Une FFT est calculée sur chaque segment fenêtré afin de transformer le signal du domaine temporel vers le domaine fréquentiel.
- Le zero-padding consiste à étendre la longueur de la séquence d’entrée (par ajout de zéros) avant la FFT.
  Cela permet d'obtenir un spectre avec une densité de points plus élevée, facilitant l'interpolation pour un raffinement de précision de la fréquence.
- On obtient un spectrogramme sous forme de matrices de fréquences, temps et amplitudes.
- Un algorithme de détection de pics identifie les fréquences où l'amplitude dépasse un certain seuil défini (AMP_THRESH).
- Pour chaque pic, une interpolation est effectuée (par exemple, quadratique sur 3 points ou par interpolation sur 5 points via régression polynomiale)
  afin d’estimer de manière plus précise la position du maximum dans le spectre.
- Ce raffinement tient compte de la résolution effective de la FFT, notamment avec le zero-padding (la conversion en fréquence se fait en considérant la taille totale nfft).
- Les indices de pics raffinés sont convertis en fréquences (Hz) en utilisant la relation f = indice raffiné × (FS / nfft).
- Ces fréquences sont ensuite converties en notes MIDI, permettant de discriminer précisément les notes entendues dans le signal.
"""

import threading
import logging
import math
from typing import List, Tuple

import numpy as np
import pyaudio
from scipy.signal import spectrogram, find_peaks, butter, filtfilt

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
import matplotlib.animation as animation

# Constantes globales
BUF_SIZE = 512
FS = 11025
AMP_THRESH = 0.5
PEAK_LIMIT = 4
NORM_FACTOR = AMP_THRESH * 32
FLAG_SHIFT = True
GRID_W = 9
GRID_H = 7
UPDATE_MSEC = 10
COLORMAP_CHOICE = "rainbow"
INTERPOLATION = "5points"
ZERO_PADDING_FACTOR = 4

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calc_pitch(i: int, j: int) -> int:
    """
    Calcule la classe de hauteur (pitch class) basée sur une combinaison linéaire des indices i et j.
    La formule (7 * i + 4 * j) % 12 permet de générer une répartition sur les 12 classes.

    Args:
        i (int): Indice sur l'axe horizontal.
        j (int): Indice sur l'axe vertical.
    
    Returns:
        int: La pitch class (entre 0 et 11).
    """
    return (7 * i + 4 * j) % 12


def pitch_label(pc: int) -> str:
    """
    Retourne une étiquette textuelle pour la pitch class sous forme hexadécimale en majuscules.

    Args:
        pc (int): La pitch class.
    
    Returns:
        str: L'étiquette de la pitch.
    """
    return hex(pc)[2:].upper()


def create_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crée une grille de pitch en fonction des dimensions GRID_W et GRID_H. 
    Calcule également les bornes en x et en y pour une utilisation en pcolormesh.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - La grille de pitch.
            - Les bornes en x.
            - Les bornes en y.
    """
    range_i = np.arange(-GRID_W // 2, GRID_W // 2)
    range_j = np.arange(-GRID_H // 2, GRID_H // 2)
    grid = np.array([[calc_pitch(i, j) for i in range_i] for j in range_j])
    x_bounds = np.arange(range_i.min() - 0.5, range_i.max() + 1, 1)
    y_bounds = np.arange(range_j.min() - 0.5, range_j.max() + 1, 1)
    return grid, x_bounds, y_bounds


def rainbow_cmap() -> mcolors.ListedColormap:
    """
    Génère une colormap basée sur une échelle de couleurs RVB issues de la conversion d'Espace HSV.

    Returns:
        mcolors.ListedColormap: La colormap rainbow.
    """
    cols = [hsv_to_rgb([i / 12, 0.5, 0.75]) for i in range(12)]
    return mcolors.ListedColormap(cols)


def piano_cmap() -> mcolors.ListedColormap:
    """
    Génère une colormap spécifique de type piano avec des teintes de gris.

    Returns:
        mcolors.ListedColormap: La colormap piano.
    """
    mapping = {(5 + n * 7) % 12: 'lightgray' if n <= 6 else 'darkgray' for n in range(12)}
    cols = [mapping[i] for i in range(12)]
    return mcolors.ListedColormap(cols)


def freq_to_midi(freq: float) -> float:
    """
    Convertit une fréquence (Hz) en note MIDI.
    
    Args:
        freq (float): La fréquence en Hz.
    
    Returns:
        float: La valeur correspondante en note MIDI.
               Retourne 0.0 si la fréquence est négative ou nulle.
    """
    if freq > 0:
        return 69 + 12 * np.log2(freq / 440.0)
    return 0.0


def butter_lowpass(cutoff: float, fs: float, order: int = 4):
    """
    Conçoit un filtre Butterworth passe-bas.

    Args:
        cutoff (float): Fréquence de coupure en Hz.
        fs (float): Fréquence d'échantillonnage en Hz.
        order (int): Ordre du filtre.
    
    Returns:
        b, a: Coefficients du filtre (pôles et zéros).
    """
    nyq = 0.5 * fs  # Fréquence de Nyquist
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_lowpass_filter(sig: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Applique un filtre passe-bas Butterworth au signal.

    Utilise filtfilt pour un filtrage zéro-phase (aucun déphasage).

    Args:
        sig (np.ndarray): Signal d'entrée.
        cutoff (float): Fréquence de coupure en Hz.
        fs (float): Fréquence d'échantillonnage en Hz.
        order (int): Ordre du filtre.

    Returns:
        np.ndarray: Signal filtré.
    """
    b, a = butter_lowpass(cutoff, fs, order)
    filtered_sig = filtfilt(b, a, sig)
    return filtered_sig


def refine_spec_3p(spec: np.ndarray, idx: int) -> float:
    """
    Raffine la position d'un pic dans le spectre par interpolation quadratique
    sur 3 points. Cela permet d'obtenir un indice (float) raffiné,
    qui sera ensuite converti en Hz via f = indice raffiné * (FS / nfft).

    Args:
        spec (np.ndarray): Spectre d'amplitudes.
        idx (int): Indice du pic détecté.
    
    Returns:
        float: Position raffinée du pic (indice flottant).
    """
    shift = 0.0
    # On vérifie que l'indice est bien au milieu du spectre afin d'éviter une erreur
    if 0 < idx < len(spec) - 1:
        A_minus = spec[idx - 1]
        A0 = spec[idx]
        A_plus = spec[idx + 1]
        # Formule d'interpolation quadratique
        denom = 2 * (A_minus - 2 * A0 + A_plus)
        if denom != 0:
            shift = (A_minus - A_plus) / denom
    return idx + shift


def refine_spec_5p(spec: np.ndarray, idx: int) -> float:
    """
    Raffine la position d'un pic dans le spectre par interpolation sur 5 points,
    en ajustant un polynôme de degré 2 (quadratique) par régression aux moindres carrés.
    
    L'intérêt est d'utiliser plus de points autour du pic pour
    potentiellement améliorer la précision de l'estimation.

    Args:
        spec (np.ndarray): Spectre d'amplitudes.
        idx (int): Indice du pic détecté.
    
    Returns:
        float: Position raffinée du pic (indice flottant).
    """
    N = len(spec)
    # Sélection de 5 points : 2 avant, l'indice central, 2 après.
    # Gestion des cas limites (débordement sur les bords)
    if idx < 2:
        indices = np.arange(0, min(5, N))
    elif idx > N - 3:
        indices = np.arange(max(0, N - 5), N)
    else:
        indices = np.arange(idx - 2, idx + 3)
    
    # Coordonnées x (indices) et amplitudes y correspondantes
    x = indices.astype(float)
    y = spec[indices]
    
    # Ajustement d'un polynôme de degré 2 sur ces points :
    # y = ax^2 + bx + c
    p = np.polyfit(x, y, 2)
    a, b, _ = p
    # La position du maximum du polynôme est donnée par -b / (2a)
    # On vérifie que a différent de 0 pour éviter la division par zéro
    if a == 0:
        refined_idx = idx
    else:
        refined_idx = -b / (2 * a)
    
    return refined_idx


def analyze_chunk(ch: bytes) -> List[Tuple[float, float]]:
    """
    Analyse un bloc de données audio (chunk) afin d’en extraire les notes et leurs amplitudes.
    
    Processus complet :
      1. Décodage du flux audio : conversion de bytes en int16, puis en float32.
      2. (Optionnel) Application d'un filtre antibruit passe-bas pour atténuer le bruit.
      3. Calcul du spectrogramme :
         - Découpage du signal en trames de taille BUF_SIZE avec un chevauchement (noverlap).
         - Application d'une fenêtre Hann à chaque trame pour limiter la fuite spectrale.
         - Utilisation du zero-padding (nfft = BUF_SIZE * ZERO_PADDING_FACTOR) pour améliorer
           la densité du spectre sans augmenter la latence temporelle.
      4. Pour chaque trame, la détection des pics se fait via find_peaks sous un seuil AMP_THRESH.
      5. Pour chaque pic détecté, un raffinement par interpolation (quadratique sur 3 points
         ou 5 points) est effectué afin d'obtenir une estimation précise de la position du pic dans le spectre.
      6. Conversion des positions raffinées (indices flottants) en fréquence en Hz.
      7. Conversion de la fréquence en note MIDI via la fonction freq_to_midi.

    Args:
        ch (bytes): Données audio brutes.

    Returns:
        List[Tuple[float, float]]: Liste de tuples (note MIDI, amplitude du pic).
    """
    # Vérification de validité du chunk
    if not ch or len(ch) < 2:
        return []
    
    # Décodage : conversion des bytes en int16 puis en float32 pour la compatibilité numérique
    sig = np.frombuffer(ch, dtype=np.int16).astype(np.float32)
    if sig.size == 0:
        return []
    
    # -------------------------------
    # Etape 1 : Filtrage antibruit
    # -------------------------------
    # Exemple : application d'un filtre passe-bas pour ne conserver que les composantes
    # en dessous de 2000 Hz. La valeur du cutoff est à adapter selon le cas.
    CUTOFF_FREQ = 2000  # en Hz, à ajuster selon vos besoins
    sig = apply_lowpass_filter(sig, cutoff=CUTOFF_FREQ, fs=FS, order=4)
    
    # ---------------------------------------------------------------------------------------
    # Etape 2 : Calcul du spectrogramme avec fenêtrage et zero-padding.
    # ---------------------------------------------------------------------------------------
    nfft = BUF_SIZE * ZERO_PADDING_FACTOR  # Zero-padding : augmentation de la taille de la FFT
    # Utilisation de la fonction spectrogram() de scipy.signal :
    # - nperseg définit la longueur de chaque segment.
    # - noverlap définit le chevauchement entre segments.
    # - nfft impose la taille de la FFT (intégrant le zero-padding).
    freqs, times, mags = spectrogram(
        sig,
        fs=FS,
        window='hann',
        nperseg=BUF_SIZE,
        noverlap=BUF_SIZE // 2,
        nfft=nfft,
        mode='magnitude'
    )
    
    results = []  # Liste qui contiendra les tuples (note MIDI, amplitude)

    # Traitement de chaque trame (chaque colonne du spectrogramme)
    for spec in mags.T:
        # Détection des pics dépassant le seuil d'amplitude AMP_THRESH
        peaks, props = find_peaks(spec, height=AMP_THRESH)
        if peaks.size == 0:
            continue

        # Sélection des PEAK_LIMIT pics les plus forts
        heights = props['peak_heights']
        sorted_idx = np.argsort(heights)[::-1][:PEAK_LIMIT]

        for s in sorted_idx:
            p = int(peaks[s])
            amp = heights[s]
            
            # Raffinement du pic :
            if INTERPOLATION == "3points":
                refined = refine_spec_3p(spec, p)
            else:
                refined = refine_spec_5p(spec, p)

            # Conversion de l'indice raffiné en fréquence (en Hz).
            # La résolution fréquentielle est FS / nfft.
            f_ref = refined * (FS / nfft)

            # Conversion de la fréquence en note MIDI
            note = freq_to_midi(f_ref)
            
            results.append((note, amp))
    
    return results


class AudioHandler:
    """
    Gère l'ouverture, la lecture et la fermeture du flux audio via pyaudio.
    """
    def __init__(self) -> None:
        self.pa = pyaudio.PyAudio()
        try:
            self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=FS,
                                       input=True, frames_per_buffer=BUF_SIZE)
        except Exception as e:
            logging.error("Erreur d'ouverture du flux audio: %s", e)
            raise

    def read_buffer(self) -> bytes:
        """
        Lit un buffer audio et le retourne sous forme de bytes.
        
        Returns:
            bytes: Le buffer audio lu.
        """
        try:
            return self.stream.read(BUF_SIZE, exception_on_overflow=False)
        except Exception as e:
            logging.error("Erreur lors de la lecture audio: %s", e)
            return b''

    def close_stream(self) -> None:
        """
        Ferme le flux audio et termine l'instance pyaudio.
        """
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()


class NoteRepository:
    """
    Repository thread-safe pour stocker et récupérer les notes détectées.
    """
    def __init__(self) -> None:
        self.notes: List[Tuple[float, float]] = []
        self.mutex = threading.Lock()
    
    def update_notes(self, notes: List[Tuple[float, float]]) -> None:
        """
        Met à jour la liste des notes détectées de manière thread-safe.
        
        Args:
            notes (List[Tuple[float, float]]): Les notes détectées à mettre à jour.
        """
        with self.mutex:
            self.notes = notes.copy()
    
    def fetch_notes(self) -> List[Tuple[float, float]]:
        """
        Récupère la liste actuelle des notes détectées de manière thread-safe.
        
        Returns:
            List[Tuple[float, float]]: La copie des notes stockées.
        """
        with self.mutex:
            return self.notes.copy()


class GridDisplay:
    """
    Affiche une représentation en grille des notes détectées sur une fenêtre matplotlib.
    """
    def __init__(self, note_repo: NoteRepository) -> None:
        self.note_repo = note_repo
        self.grid, self.x_bounds, self.y_bounds = create_grid()
        if COLORMAP_CHOICE == "piano":
            self.cmap = piano_cmap()
        elif COLORMAP_CHOICE == "rainbow":
            self.cmap = rainbow_cmap()
        self.positions: dict[int, List[Tuple[float, float]]] = {}
        num_rows, num_cols = self.grid.shape
        x_vals = np.linspace(-num_cols // 2, num_cols // 2 - 1, num_cols)
        y_vals = np.linspace(-num_rows // 2, num_rows // 2 - 1, num_rows)
        for j, y in enumerate(y_vals):
            for i, x in enumerate(x_vals):
                pc = self.grid[j, i]
                self.positions.setdefault(pc, []).append((x, y))
    
    def refresh(self, ax: plt.Axes) -> None:
        """
        Met à jour l'affichage de la grille en dessinant le fond, les étiquettes et les notes
        avec leurs amplitudes.
        
        Args:
            ax (plt.Axes): L'axe matplotlib à rafraîchir.
        """
        nts = self.note_repo.fetch_notes()
        ax.clear()
        ax.set_aspect('equal')
        ax.axis('off')
        ax.pcolormesh(self.x_bounds, self.y_bounds, self.grid, cmap=self.cmap, edgecolors='black')
        # Affichage des labels correspondants aux pitch classes
        for pc, pts in self.positions.items():
            for (x, y) in pts:
                ax.text(x, y, pitch_label(pc), ha='center', va='center', color='black', fontsize=12)
        # Affichage des notes détectées avec ajustement de position selon le décalage (cents_shift)
        for note_val, amp in nts:
            target_pc = int(round(note_val)) % 12
            cents_shift = (note_val - round(note_val)) * 100
            delta = - (cents_shift / 50) * 0.5 if FLAG_SHIFT else 0
            alpha = min(0.75, amp / NORM_FACTOR)
            for (x, y) in self.positions.get(target_pc, []):
                ax.scatter(x + delta, y + delta, s=500, marker='+', color='white', alpha=alpha, lw=4)


class CircularDisplay:
    """
    Affiche une représentation circulaire (wedge) des notes détectées sur une fenêtre matplotlib.
    """
    def __init__(self, note_repo: NoteRepository) -> None:
        self.note_repo = note_repo
        self.radius = 1.0
        self.col = [hsv_to_rgb([hue, 0.5, 0.75]) for hue in np.linspace(0, 1, 12, endpoint=False)]
        self.wedge_params = []
        for st in range(12):
            angle_start = 105 - (st * 360 / 12)
            angle_end   = 105 - ((st + 1) * 360 / 12)
            ang = (math.pi / 2) - (2 * math.pi * st / 12)
            x_lab = (self.radius * 5/6) * math.cos(ang)
            y_lab = (self.radius * 5/6) * math.sin(ang)
            mid_ang = (math.pi / 2) - (2 * math.pi * (st + 0.5) / 12)
            x_end = self.radius * math.cos(mid_ang)
            y_end = self.radius * math.sin(mid_ang)
            self.wedge_params.append((angle_start, angle_end, self.col[st], x_lab, y_lab, x_end, y_end))
    
    def refresh(self, ax: plt.Axes) -> None:
        """
        Met à jour l'affichage circulaire en dessinant le cercle, les wedges de couleur
        et la représentation des notes détectées.
        
        Args:
            ax (plt.Axes): L'axe matplotlib à rafraîchir.
        """
        ax.clear()
        ax.set_aspect('equal')
        ax.axis('off')
        # Dessine le cercle principal
        circ = plt.Circle((0, 0), self.radius, lw=4, edgecolor='black', facecolor='none')
        ax.add_artist(circ)
        ax.set_xlim(-self.radius * 6/5, self.radius * 6/5)
        ax.set_ylim(-self.radius * 6/5, self.radius * 6/5)
        # Dessine les segments de couleurs (wedge) et ajoute les labels
        for angle_start, angle_end, facecolor, x_lab, y_lab, x_end, y_end in self.wedge_params:
            wedge = Wedge(center=(0, 0), r=self.radius, theta1=angle_end, theta2=angle_start, facecolor=facecolor)
            ax.add_patch(wedge)
            st = self.wedge_params.index((angle_start, angle_end, facecolor, x_lab, y_lab, x_end, y_end))
            ax.text(x_lab, y_lab, pitch_label(st), ha='center', va='center', color='black', fontsize=12)
            ax.plot([0, x_end], [0, y_end], color='black')
        # Affichage des notes détectées sous forme de lignes allant du centre au bord du cercle
        nts = self.note_repo.fetch_notes()
        for note_val, amp in nts:
            ang = (math.pi / 2) - (2 * math.pi * (note_val % 12) / 12)
            x_pt = self.radius * math.cos(ang)
            y_pt = self.radius * math.sin(ang)
            a_val = min(0.75, amp / NORM_FACTOR)
            ax.plot([0, x_pt], [0, y_pt], color='white', alpha=a_val, lw=4)


def audio_worker(note_repo: NoteRepository, stop_evt: threading.Event) -> None:
    """
    Fonction exécutée dans un thread dédié à la lecture audio.
    Elle lit continuellement des blocs audio, les analyse pour extraire des notes,
    et met à jour le repository de notes.
    
    Args:
        note_repo (NoteRepository): Repository pour stocker les notes détectées.
        stop_evt (threading.Event): Event pour signaler l'arrêt de la boucle.
    """
    audio = AudioHandler()
    try:
        while not stop_evt.is_set():
            chunk = audio.read_buffer()
            notes = analyze_chunk(chunk)
            note_repo.update_notes(notes)
    except Exception as ex:
        logging.error("Erreur audio: %s", ex)
    finally:
        audio.close_stream()


class Animator:
    """
    Gère l'animation des affichages GridDisplay et CircularDisplay sur des fenêtres matplotlib.
    """
    def __init__(self, note_repo, grid_disp, circ_disp):
        self.note_repo = note_repo
        self.grid_disp = grid_disp
        self.circ_disp = circ_disp
        self.fig1, self.ax1 = plt.subplots(figsize=(GRID_W, GRID_H), facecolor='gray')
        self.fig2, self.ax2 = plt.subplots(figsize=(7, 7), facecolor='gray')
        self.fig1.canvas.manager.set_window_title("Grid Display")
        self.fig2.canvas.manager.set_window_title("Circular Display")
        try:
            mgr1 = self.fig1.canvas.manager
            mgr2 = self.fig2.canvas.manager
            mgr1.window.wm_geometry("+0+100")
            mgr2.window.wm_geometry("+950+100")
        except Exception as e:
            logging.warning("Impossible de positionner les fenêtres automatiquement: %s", e)

    def animate(self, frame):
        """
        Fonction d'animation appelée périodiquement par matplotlib.
        Rafraîchit les deux affichages.
        """
        self.grid_disp.refresh(self.ax1)
        self.circ_disp.refresh(self.ax2)
        self.fig1.canvas.draw_idle()
        self.fig2.canvas.draw_idle()

    def show(self):
        """
        Lance l'animation et affiche les fenêtres matplotlib.
        """
        ani = animation.FuncAnimation(self.fig1, self.animate, interval=UPDATE_MSEC, cache_frame_data=False)
        plt.show()


def main() -> None:
    """
    Point d'entrée principal du script.
    Initialise le repository de notes, les affichages, démarre le thread audio,
    et lance l'animation.
    """
    note_repo = NoteRepository()
    grid_disp = GridDisplay(note_repo)
    circ_disp = CircularDisplay(note_repo)
    stop_evt = threading.Event()
    audio_thread = threading.Thread(target=audio_worker, args=(note_repo, stop_evt), daemon=True)
    audio_thread.start()
    animator = Animator(note_repo, grid_disp, circ_disp)
    animator.show()
    stop_evt.set()


if __name__ == '__main__':
    main()
