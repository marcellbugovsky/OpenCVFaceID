# OpenCVFaceID

Ein Echtzeit-Gesichtserkennungssystem, das Python und das DNN-Modul von OpenCV verwendet. Es erkennt Gesichter in einem Live-Webcam-Feed und identifiziert bekannte Personen anhand einer vorab erstellten Datenbank von Gesichtsmerkmalen (Embeddings).

## Projektbeschreibung

Dieses Projekt implementiert eine Pipeline zur Gesichtserkennung in Echtzeit:

1.  **Datenbankerstellung:** Es scannt ein Verzeichnis mit Bildern bekannter Personen (`known_faces`), erkennt Gesichter darin, extrahiert eindeutige Merkmalsvektoren (Embeddings) mit einem tiefen neuronalen Netzwerk (DNN) und speichert diese zusammen mit den Namen in einer Datenbank (`.pkl`-Datei).
2.  **Live-Erkennung:** Es greift auf eine Webcam zu, erkennt Gesichter in jedem Frame mithilfe eines DNN-basierten Detektors, extrahiert die Embeddings der erkannten Gesichter und vergleicht diese mit den Embeddings in der Datenbank. Bei ausreichender Ähnlichkeit (geringer euklidischer Abstand) wird der Name der bekannten Person im Videostream angezeigt.

## Features

* **Echtzeit-Verarbeitung:** Führt Gesichtserkennung live über einen Webcam-Feed durch.
* **DNN-basierte Modelle:** Nutzt vortrainierte Deep Learning-Modelle via OpenCVs DNN-Modul für robuste Gesichtsdetektion (Caffe ResNet10-SSD) und -erkennung (ONNX SFace).
* **Datenbank für bekannte Gesichter:** Ermöglicht das einfache Hinzufügen neuer Personen durch Erstellen von Ordnern und Hinzufügen von Bildern. Ein Skript (`build_database.py`) automatisiert die Erstellung der Embedding-Datenbank.
* **Konfigurierbar:** Viele Parameter wie Pfade, Kamera-Index, Modellparameter und Erkennungsschwellenwerte können über eine `config.yaml`-Datei angepasst werden.
* **Modulare Struktur:** Code für Kamera, Detektion, Enkodierung und Konfiguration ist in separaten Modulen im `src`-Verzeichnis organisiert.

## Verwendete Technologien

* **Sprache:** Python 3
* **Kernbibliotheken:**
    * OpenCV (`opencv-python`): Für Bild-/Videoverarbeitung, DNN-Modul, Kamera-Zugriff.
    * NumPy (`numpy`): Für numerische Operationen, insbesondere Vektorberechnungen (Embeddings, Distanzen).
    * PyYAML (`PyYAML`): Zum Laden der Konfigurationsdatei (`config.yaml`).
    * *Optional:* `scikit-learn` (in `requirements.txt` gelistet, aber die Kernfunktionalität scheint `numpy.linalg.norm` zu verwenden).

## Verwendete Modelle

* **Gesichtsdetektion:** OpenCV DNN mit Caffe ResNet10-SSD
    * `deploy.prototxt`
    * `res10_300x300_ssd_iter_140000.caffemodel`
* **Gesichtserkennung (Encoding):** OpenCV DNN mit ONNX SFace
    * `face_recognition_sface_2021dec.onnx`

*(Hinweis: Diese Modelldateien sind nicht Teil dieses Repositories und müssen separat heruntergeladen und im `models`-Verzeichnis platziert werden.)*

## Setup & Installation

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/marcellbugovsky/OpenCVFaceID.git](https://github.com/marcellbugovsky/OpenCVFaceID.git)
    cd OpenCVFaceID
    ```
2.  **Modelle herunterladen:**
    * Lade die benötigten Modell-Dateien herunter (siehe Abschnitt "Verwendete Modelle").
    * Erstelle das Verzeichnis `models/detection/` und platziere `deploy.prototxt` sowie `res10_300x300_ssd_iter_140000.caffemodel` darin.
    * Erstelle das Verzeichnis `models/recognition/` und platziere `face_recognition_sface_2021dec.onnx` darin.
    *(Du solltest Links oder Quellen für diese Modelle angeben, falls möglich.)*

3.  **Bekannte Gesichter hinzufügen:**
    * Erstelle das Verzeichnis `known_faces/`.
    * Erstelle für jede Person, die erkannt werden soll, einen Unterordner mit dem Namen der Person (z.B. `known_faces/Max_Mustermann/`).
    * Platziere mehrere Bilder (JPG, PNG) dieser Person in ihrem jeweiligen Ordner. Stelle sicher, dass das Gesicht auf den Bildern gut sichtbar ist.

4.  **Virtuelle Umgebung erstellen (empfohlen):**
    ```bash
    python -m venv venv
    venv\Scripts\activate    # Windows
    ```
5.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Konfiguration überprüfen:**
    * Öffne `config/config.yaml`. Überprüfe die Pfade (`known_faces_dir`, `models_dir`, `database_dir`, Modellpfade) und den `camera_index` (oft 0 oder 1). Passe den `recognition_threshold_distance` bei Bedarf an (niedrigere Werte sind strenger).

## Verwendung

**Wichtiger Hinweis:** Führe die Skripte aus dem Hauptverzeichnis des Projekts (`OpenCVFaceID/`) aus.

1.  **Datenbank erstellen:**
    * Dieses Skript verarbeitet die Bilder im `known_faces`-Verzeichnis, extrahiert Embeddings und speichert sie. Führe dies einmalig aus und immer dann, wenn du Personen/Bilder im `known_faces`-Verzeichnis hinzufügst oder änderst.
    ```bash
    python build_database.py
    ```

2.  **Live-Erkennung starten:**
    * Startet den Webcam-Feed, erkennt Gesichter und versucht, sie anhand der erstellten Datenbank zu identifizieren.
    ```bash
    python run_live_recognition.py
    ```
    * Drücke 'q', um die Anwendung zu beenden.

## Konfiguration (`config/config.yaml`)

Die `config.yaml`-Datei steuert verschiedene Aspekte:

* **Pfade:** Speicherorte für bekannte Gesichter, Modelle und die Embedding-Datenbank.
* **Kamera:** Index der zu verwendenden Webcam.
* **Detektor:** Pfade zu den Detektormodellen, Konfidenzschwelle, Eingabegröße, Mittelwertsubtraktion.
* **Recognizer:** Pfad zum Erkennungsmodell (ONNX), erwartete Eingabegröße, Skalierungsfaktor, Kanalreihenfolge (RGB/BGR).
* **Erkennungsschwelle:** Maximal erlaubter Abstand (Euclidean Distance), damit ein Gesicht als bekannt erkannt wird. Muss eventuell angepasst werden.
* **UI:** Fenstertitel, Farbe der Bounding Box, Bezeichnung für unbekannte Personen.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz.
