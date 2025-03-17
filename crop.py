import cv2
import numpy as np, subprocess as sp, json
from customFunction import checkCam, closeCam

# Variabili globali per le coordinate del mouse
mouse_x1, mouse_y1 = -1, -1
mouse_x2, mouse_y2 = -1, -1
is_x1_filled = False

output_file = "/home/ae/VBOM-PROJECT/crop.txt"
json_file = "/home/ae/VBOM-Project/PARAMETERS.json"

def leggi_filetxt(file_path):
    # Legge il file e restituisce una lista di liste con i dati
    dati = []
    with open(file_path, 'r') as file:
        for line in file:
            dati.append(list(map(int, line.strip().split(', '))))
    return dati

def scrivi_filetxt(file_path, dati):
    # Scrive i dati aggiornati nel file
    with open(file_path, 'w') as file:
        for riga in dati:
            file.write(', '.join(map(str, riga)) + '\n')

def leggi_dati_json(file_path):
    
   """Legge il file JSON e restituisce i dati come dizionario."""
   try:
    with open(file_path, 'r') as file:
        return json.load(file)
   except json.JSONDecodeError as e:
    print(f"Errore nel file JSON: {e}. Rigenero il file...")
    # Rigenera un file JSON valido
    dati_iniziali = {
        "crop_values": {}
    }
    with open(file_path, 'w') as file:
        json.dump(dati_iniziali, file, indent=4)
    return dati_iniziali

def writeOnJson(file_path, dati):
    """Scrive i dati nel file JSON."""
    with open(file_path, 'w') as file:
        json.dump(dati, file, indent=4, separators=(',', ': '))

def upgradeCropValues(file_path, camera_id, valori_crop):
    """Aggiorna i valori di crop nel file JSON."""
    dati = leggi_dati_json(file_path)
    dati["crop_values"][str(camera_id)] = valori_crop
    writeOnJson(file_path, dati)


def mouse_callback1(event, x1, y1, flags, param):
    global mouse_x1, mouse_y1, is_x1_filled
    
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x1, mouse_y1 = x1, y1
    
    elif event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinate: ({mouse_x1}, {mouse_y1})")
        
        dati = leggi_dati_json(json_file)
        crop_values = dati.get("crop_values", {})
        
        if str(WEBCAM) in crop_values:
            if not is_x1_filled: 
                crop_values[str(WEBCAM)][0] = mouse_x1
                crop_values[str(WEBCAM)][1] = mouse_y1
                is_x1_filled = True
            else:
                crop_values[str(WEBCAM)][2] = mouse_x1
                crop_values[str(WEBCAM)][3] = mouse_y1
                is_x1_filled = False
        else:
            crop_values[str(WEBCAM)] = [mouse_x1, mouse_y1, 0, 0]  # Valori di default
        
        dati["crop_values"] = crop_values
        writeOnJson(json_file, dati)


video_available = checkCam()

print("Webcam disponibili: ", video_available)
WEBCAM = int(input("Flusso video:"))

# Inizializza la webcam
cap1 = cv2.VideoCapture(WEBCAM)
cap1.set(3, 1920)
cap1.set(4, 1080)

# Crea una finestra e imposta la funzione di callback del mouse
cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", mouse_callback1)


# Ciclo principale del programma
while True:
    
    # Leggi un frame dalla webcam
    ret, frame1 = cap1.read()

    # Visualizza l'immagine nella finestra
    cv2.imshow("Webcam", frame1)

    # Esci dal ciclo quando viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'): break


closeCam()
# Rilascia la webcam e chiudi la finestra
cap1.release()
cv2.destroyAllWindows()
