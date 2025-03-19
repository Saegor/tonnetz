#!/usr/bin/env python3
"""
Application d'analyse tonale qui capte de l'audio, extrait les notes
et les affiche sur une grille ainsi qu'une vue circulaire.
"""
import math
import threading
import logging
from typing import List, Tuple, Dict

import pyaudio
import numpy as np
from scipy.signal import find_peaks, spectrogram
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Wedge

# Type alias pour une note : (valeur MIDI, amplitude)
Note = Tuple[float, float]

# Constantes pour l'audio et l'affichage
AUDIO_BUFFER_SIZE: int = 1024
SAMPLE_RATE: int = 44100
AMPLITUDE_THRESHOLD: float = 1.0
NORMALIZATION_FACTOR: float = AMPLITUDE_THRESHOLD * 16
PEAK_LIMIT: int = 4
GRID_WIDTH: int = 9
GRID_HEIGHT: int = 7

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def process_audio_buffer(audio_chunk: bytes) -> List[Note]:
    """
    Traite un tampon audio et retourne une liste de notes détectées.

    Args:
        audio_chunk: Données audio brutes.

    Returns:
        Liste de tuples contenant (note MIDI, amplitude).
    """
    if not audio_chunk or len(audio_chunk) < 2:
        return []

    # Conversion du buffer audio en tableau numpy pour traitement
    signal: np.ndarray = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
    if signal.size == 0:
        return []

    # Calcul du spectrogramme
    _, _, magnitude_spectrogram = spectrogram(
        signal,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=AUDIO_BUFFER_SIZE,
        noverlap=AUDIO_BUFFER_SIZE // 2,
        mode="magnitude",
    )

    notes_detected: List[Note] = []
    # Parcours de chaque colonne du spectrogramme (par fenêtre temporelle)
    for spectrum in magnitude_spectrogram.T:
        peaks, properties = find_peaks(spectrum, height=AMPLITUDE_THRESHOLD)
        if peaks.size == 0:
            continue

        peak_heights: np.ndarray = properties["peak_heights"]
        # Tri décroissant des amplitudes et limitation du nombre de pics considérés
        sorted_indices: np.ndarray = np.argsort(peak_heights)[::-1][:PEAK_LIMIT]
        for sorted_idx in sorted_indices:

            peak_index: int = int(peaks[sorted_idx])
            amplitude: float = peak_heights[sorted_idx]
            
            # interpolation parabolique
            shift: float = 0.0
            if 0 < peak_index < len(spectrum) - 1:
                amp_minus: float = spectrum[peak_index - 1]
                amp_center: float = spectrum[peak_index]
                amp_plus: float = spectrum[peak_index + 1]
                denominator: float = 2 * (amp_minus - 2 * amp_center + amp_plus)
                if denominator != 0:
                    shift = (amp_minus - amp_plus) / denominator

            frequency: float = (peak_index + shift) * (SAMPLE_RATE / AUDIO_BUFFER_SIZE)
            note_midi: float = (
                69 + 12 * np.log2(frequency / 440.0)
                if frequency > 0 else 0.0
            )
            notes_detected.append((note_midi, amplitude))
    return notes_detected


class AudioStreamHandler:
    """
    Gère la réception du flux audio en utilisant PyAudio.
    """

    def __init__(self) -> None:
        self.py_audio: pyaudio.PyAudio = pyaudio.PyAudio()
        try:
            self.stream: pyaudio.Stream = self.py_audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=AUDIO_BUFFER_SIZE,
            )
        except Exception as error:
            logging.error("Erreur lors de l'ouverture du flux audio: %s", error)
            raise

    def __enter__(self) -> "AudioStreamHandler":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close_stream()

    def read_audio_chunk(self) -> bytes:
        """
        Lit et retourne un tampon audio.

        Returns:
            Un bloc de données audio.
        """
        try:
            return self.stream.read(AUDIO_BUFFER_SIZE,
                                    exception_on_overflow=False)
        except Exception as error:
            logging.error("Erreur lors de la lecture audio: %s", error)
            return b""

    def close_stream(self) -> None:
        """Arrête et ferme le flux audio."""
        self.stream.stop_stream()
        self.stream.close()
        self.py_audio.terminate()


class NoteRepository:
    """
    Dépôt de notes détectées avec accès thread-safe.
    """

    def __init__(self) -> None:
        self._notes: List[Note] = []
        self._lock: threading.Lock = threading.Lock()

    def update_notes(self, notes: List[Note]) -> None:
        """
        Met à jour les notes détectées dans le dépôt.

        Args:
            notes: Liste de notes à stocker.
        """
        with self._lock:
            self._notes = notes.copy()

    def get_notes(self) -> List[Note]:
        """
        Retourne une copie des dernières notes détectées.

        Returns:
            Liste contenant la dernière mise à jour des notes.
        """
        with self._lock:
            return self._notes.copy()


class GridVisualizer:
    """
    Affiche une grille de pitchs et les notes sur cette grille.
    """

    def __init__(self, note_repo: NoteRepository) -> None:
        self.note_repo: NoteRepository = note_repo
        x_range: np.ndarray = np.arange(0, GRID_WIDTH)
        y_range: np.ndarray = np.arange(0, GRID_HEIGHT)
        # Construction d'une grille harmonique
        self.pitch_grid: np.ndarray = np.array(
            [[(7 * x + 4 * y) % 12 for x in x_range] for y in y_range]
        )
        self.x_bounds: np.ndarray = np.arange(x_range.min() - 0.5,
                                              x_range.max() + 1, 1)
        self.y_bounds: np.ndarray = np.arange(y_range.min() - 0.5,
                                              y_range.max() + 1, 1)
        # Génération de la palette de couleurs pour les 12 classes de pitch
        colors = [hsv_to_rgb([i / 12, 0.5, 0.75]) for i in range(12)]
        self.colormap: mcolors.ListedColormap = mcolors.ListedColormap(colors)

        num_rows, num_cols = self.pitch_grid.shape
        x_positions: List[float] = list(np.linspace(0, num_cols - 1, num_cols))
        y_positions: List[float] = list(np.linspace(0, num_rows - 1, num_rows))
        # Dictionnaire associant à chaque pitch les positions sur la grille
        self.pitch_positions: Dict[int, List[Tuple[float, float]]] = {
            self.pitch_grid[row, col]: []
            for row in range(num_rows)
            for col in range(num_cols)
        }
        for row, y in enumerate(y_positions):
            for col, x in enumerate(x_positions):
                pitch_class = self.pitch_grid[row, col]
                self.pitch_positions[pitch_class].append((x, y))

    def draw_grid(self, ax: plt.Axes) -> None:
        """
        Dessine la grille et ajoute les étiquettes de pitch à chaque position.
        """
        ax.pcolormesh(self.x_bounds, self.y_bounds, self.pitch_grid,
                      cmap=self.colormap, edgecolors="black")
        for pitch_class, points in self.pitch_positions.items():
            for (x, y) in points:
                ax.text(x, y, hex(pitch_class)[2:].upper(),
                        ha="center", va="center", color="black", fontsize=12)

    def draw_notes(self, ax: plt.Axes, notes: List[Note]) -> None:
        """
        Affiche les notes détectées sur la grille à l'aide de marqueurs.

        Args:
            ax: Axes de matplotlib où dessiner.
            notes: Liste des notes détectées.
        """
        for note_value, amplitude in notes:
            target_pitch: int = int(round(note_value)) % 12
            cents_shift: float = (note_value - round(note_value)) * 100
            offset: float = - (cents_shift / 50) * 0.5
            alpha: float = min(0.75, amplitude / NORMALIZATION_FACTOR)
            for (x, y) in self.pitch_positions.get(target_pitch, []):
                ax.scatter(x + offset, y + offset, s=500, marker="+",
                           color="white", alpha=alpha, lw=4)

    def refresh(self, ax: plt.Axes) -> None:
        """
        Actualise l'affichage en redessinant la grille et les notes.
        """
        notes = self.note_repo.get_notes()
        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(self.x_bounds[0] - 0.25, self.x_bounds[-1] + 0.25)
        ax.set_ylim(self.y_bounds[0] - 0.25, self.y_bounds[-1] + 0.25)
        ax.axis("off")
        self.draw_grid(ax)
        self.draw_notes(ax, notes)


class CircularVisualizer:
    """
    Affiche les notes autour d'un cercle avec des secteurs colorés.
    """

    def __init__(self, note_repo: NoteRepository) -> None:
        self.note_repo: NoteRepository = note_repo
        self.radius: float = 1.0
        self.colors = [hsv_to_rgb([hue, 0.5, 0.75])
                       for hue in np.linspace(0, 1, 12, endpoint=False)]
        # Pré-calcul des paramètres pour dessiner les secteurs
        self.wedge_parameters: List[
            Tuple[float, float, Tuple[float, float, float],
                  float, float, float, float]
        ] = []
        for pitch_class in range(12):
            angle_start: float = 90 + 15 - (pitch_class * 360 / 12)
            angle_end: float = 90 + 15 - ((pitch_class + 1) * 360 / 12)
            angle_rad: float = (math.pi / 2) - (2 * math.pi * pitch_class / 12)
            label_x: float = (self.radius * 5 / 6) * math.cos(angle_rad)
            label_y: float = (self.radius * 5 / 6) * math.sin(angle_rad)
            mid_angle_rad: float = (math.pi / 2) - (2 * math.pi * (pitch_class + 0.5)
                                                   / 12)
            end_x: float = self.radius * math.cos(mid_angle_rad)
            end_y: float = self.radius * math.sin(mid_angle_rad)
            self.wedge_parameters.append(
                (angle_start, angle_end, self.colors[pitch_class],
                 label_x, label_y, end_x, end_y)
            )

    def draw_circle(self, ax: plt.Axes) -> None:
        """
        Dessine le cercle extérieur.
        """
        circle = plt.Circle((0, 0), self.radius, lw=4,
                            edgecolor="black")
        ax.add_artist(circle)
        ax.set_xlim(-self.radius - 0.25, self.radius + 0.25)
        ax.set_ylim(-self.radius - 0.25, self.radius + 0.25)

    def draw_wedges(self, ax: plt.Axes) -> None:
        """
        Trace les secteurs colorés et ajoute les étiquettes de pitch.
        """
        for idx, (angle_start, angle_end, facecolor,
                  label_x, label_y, end_x, end_y) in enumerate(self.wedge_parameters):
            wedge = Wedge(
                center=(0, 0), r=self.radius,
                theta1=angle_end, theta2=angle_start,
                facecolor=facecolor
            )
            ax.add_patch(wedge)
            ax.text(label_x, label_y, hex(idx)[2:].upper(),
                    ha="center", va="center", color="black", fontsize=12)
            ax.plot([0, end_x], [0, end_y], color="black")

    def draw_notes(self, ax: plt.Axes, notes: List[Note]) -> None:
        """
        Dessine des lignes reliant le centre aux points indiquant les notes détectées.

        Args:
            ax: Axes où dessiner.
            notes: Liste des notes.
        """
        for note_value, amplitude in notes:
            angle_rad: float = (math.pi / 2) - (2 * math.pi * (note_value % 12) / 12)
            point_x: float = self.radius * math.cos(angle_rad)
            point_y: float = self.radius * math.sin(angle_rad)
            alpha: float = min(0.75, amplitude / NORMALIZATION_FACTOR)
            ax.plot([0, point_x], [0, point_y],
                    color="white", alpha=alpha, lw=4)

    def refresh(self, ax: plt.Axes) -> None:
        """
        Actualise l'affichage circulaire en redessinant le cercle, les secteurs et
        les notes.
        """
        notes = self.note_repo.get_notes()
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")
        self.draw_circle(ax)
        self.draw_wedges(ax)
        self.draw_notes(ax, notes)


class Animator:
    """
    Anime simultanément l'affichage de la grille et de la vue circulaire.
    """

    def __init__(self, note_repo: NoteRepository,
                 grid_visualizer: GridVisualizer,
                 circular_visualizer: CircularVisualizer) -> None:
        self.note_repo: NoteRepository = note_repo
        self.grid_visualizer: GridVisualizer = grid_visualizer
        self.circular_visualizer: CircularVisualizer = circular_visualizer
        self.fig, (self.ax_grid, self.ax_circular) = plt.subplots(
            1, 2,
            figsize=(GRID_WIDTH + GRID_HEIGHT, GRID_HEIGHT),
            facecolor="gray"
        )
        self.fig.canvas.manager.set_window_title("Analyse tonale")

    def animate(self, frame: int) -> None:
        """
        Fonction d'animation appelée par matplotlib à chaque itération.
        """
        self.grid_visualizer.refresh(self.ax_grid)
        self.circular_visualizer.refresh(self.ax_circular)
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        """
        Lance l'animation et affiche la fenêtre graphique.
        """
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
                                  wspace=0, hspace=0)
        ani = animation.FuncAnimation(
            self.fig, self.animate, interval=20, cache_frame_data=False
        )
        plt.show()
        plt.close(self.fig)


class Application:
    """
    Classe principale regroupant les composants de l'application.
    """

    def __init__(self) -> None:
        self.note_repo: NoteRepository = NoteRepository()
        self.grid_visualizer: GridVisualizer = GridVisualizer(self.note_repo)
        self.circular_visualizer: CircularVisualizer = CircularVisualizer(self.note_repo)
        self.animator: Animator = Animator(
            self.note_repo, self.grid_visualizer, self.circular_visualizer
        )
        self.stop_event: threading.Event = threading.Event()
        self.audio_thread: threading.Thread = threading.Thread(
            target=audio_capture_worker,
            args=(self.note_repo, self.stop_event),
            daemon=True
        )

    def start_audio(self) -> None:
        """
        Démarre le thread de capture audio.
        """
        logging.info("Démarrage du thread audio")
        self.audio_thread.start()

    def start_animation(self) -> None:
        """
        Démarre l'animation graphique.
        """
        logging.info("Démarrage de l'animation")
        self.animator.show()

    def stop(self) -> None:
        """
        Arrête l'application et attend la fin du thread audio.
        """
        logging.info("Arrêt de l'application")
        self.stop_event.set()
        self.audio_thread.join()


def audio_capture_worker(note_repo: NoteRepository,
                         stop_event: threading.Event) -> None:
    """
    Fonction worker qui capture le son et met à jour le dépôt de notes.

    Args:
        note_repo: Instance de NoteRepository pour stocker les notes.
        stop_event: Événement indiquant l'arrêt de la capture.
    """
    with AudioStreamHandler() as audio_handler:
        try:
            # Boucle de capture : vérifie régulièrement stop_event pour l'arrêt
            while not stop_event.is_set():
                audio_chunk: bytes = audio_handler.read_audio_chunk()
                if audio_chunk:
                    notes_detected = process_audio_buffer(audio_chunk)
                    note_repo.update_notes(notes_detected)
        except Exception as error:
            logging.error("Erreur audio: %s", error)


def main() -> None:
    """
    Fonction principale qui lance l'application.
    """
    app = Application()
    try:
        app.start_audio()
        app.start_animation()
    except KeyboardInterrupt:
        logging.info("Arrêt via KeyboardInterrupt")
    finally:
        app.stop()


if __name__ == "__main__":
    main()
