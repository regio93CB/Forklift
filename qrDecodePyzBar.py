import cv2
from pyzbar.pyzbar import decode, ZBarSymbol

def decode_allCode(frame):
    # Converti il frame in scala di grigi per una migliore elaborazione dei contrasti
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Applica il thresholding (Otsu) per evidenziare i contorni del QR code
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Decodifica limitando la ricerca ai QR code
    # barcodes = decode(thresh, symbols=[ZBarSymbol.QRCODE])
    # barcodes = decode(thresh, symbols=[ZBarSymbol.CODE128])
    barcodes = decode(thresh)
    
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        # Disegna un rettangolo attorno al QR code rilevato
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Decodifica il contenuto del QR code
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type  # dovrebbe essere "QRCODE"
        
        # Prepara il testo da visualizzare sul frame
        text = f"{barcode_data} ({barcode_type})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        
        print(f"QR Code rilevato: {barcode_data} | Tipo: {barcode_type}")
    
    return frame

def main():
    # Apri la sorgente video (qui la webcam di default)
    # cap = cv2.VideoCapture(0) #logi
    cap = cv2.VideoCapture(4) #realsense
    
    if not cap.isOpened():
        print("Errore: impossibile aprire il flusso video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Errore: impossibile leggere il frame.")
            break

        # Elabora il frame per rilevare i QR code
        frame = decode_allCode(frame)

        # Visualizza il frame con le annotazioni
        cv2.imshow("QR Code Scanner", frame)

        # Esce dal loop se viene premuto il tasto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
