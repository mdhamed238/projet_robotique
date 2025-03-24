import socket
import cv2
import numpy as np
import struct
import threading
import select

class Frame:
    def __init__(self,w,h):
        self.mat = np.zeros((h,w,3),dtype=np.uint8)
        self.id = -1

def incomingFrame(frame,iframe, frame_id):
    iframe.mat = frame.copy()
    iframe.id = frame_id

def gotFullData(data_buffer,iframe, frame_id):
    # Reconstruct the full frame
    full_data = b''.join([data_buffer[i] for i in sorted(data_buffer)])
    messageLength = struct.unpack("I",full_data[0:4])[0]
    frame_data = full_data[4:]
    if len(frame_data) == messageLength:
        #do we have all data
        frame_buffer = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_buffer,1)
        if frame is not None:
            # is frame ok
            incomingFrame(frame,iframe, frame_id)

def captureThread(sock,frame,stopThread,threadRunning):
    MaximumPacketSize = 1400
    timeout_ms = 0.01
    data_buffer = {}
    current_frame_id = -1
    threadRunning.set()
    while not stopThread.is_set():
        try:
            read_ready, _, _ = select.select([sock], [], [], timeout_ms)
            readSet = bool(read_ready)  # True if data is ready to be read
            if read_ready and readSet:
                packet, addr = sock.recvfrom(MaximumPacketSize)
                packet_id, frame_id = struct.unpack('II', packet[:8])
                payload = packet[8:]
                if frame_id != current_frame_id:
                    if current_frame_id != -1:
                        gotFullData(data_buffer,frame, frame_id)
                    # Reset buffer for new frame
                    data_buffer = {}
                    current_frame_id = frame_id
                data_buffer[packet_id] = payload
        except socket.error:
            continue
    threadRunning.clear()
    
# Teste robots & balles
def detect_objects(frame):
    # Convert to HSV for better color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV thresholds for RED balls
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Define HSV thresholds for BLUE balls
    lower_blue = np.array([95, 100, 100])
    upper_blue = np.array([115, 255, 255])




    # Create masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Optional: Morphological operations to clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

    # Find contours for RED balls
    red_positions = []
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if 5 < area < 100:  # Adjust these values based on ball size
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            red_positions.append((int(x), int(y)))

    # Find contours for BLUE balls
    blue_positions = []
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_blue:
        area = cv2.contourArea(cnt)
        if 5 < area < 100:  # Adjust these values based on ball size
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            blue_positions.append((int(x), int(y)))

    # We are not detecting robots here
    robot_positions = []

    return red_positions, blue_positions, robot_positions


def find_positions(mask, label):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    positions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # Seuil pour filtrer le bruit
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                positions.append((cx, cy))
    return positions


def main():
    stopProgram = threading.Event()
    stopThread = threading.Event()
    threadRunning = threading.Event()

    # Configuration réseau
    ip = ""  # écoute sur toutes les interfaces
    # ip = "192.168.1.181"  # écouter sur une interface spécifique si besoin
    port = 8080

    # Création de la socket UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    sock.bind((ip, port))

    print("Listening for UDP frames...")

    # Initialisation de l'objet Frame
    frame = Frame(100, 100)

    stopProgram.clear()
    stopThread.clear()

    # Lancement du thread de réception vidéo
    thread = threading.Thread(
        target=captureThread,
        args=[sock, frame, stopThread, threadRunning],
        daemon=True
    )
    thread.start()

    # Boucle principale
    while not stopProgram.is_set():
        if frame is not None:
            display_frame = frame.mat.copy()

            # Détection des objets sur la frame reçue
            red_positions, blue_positions, robot_positions = detect_objects(display_frame)

            # Affichage des cercles et des labels sur les objets détectés
            for pos in red_positions:
                cv2.circle(display_frame, pos, 10, (0, 0, 255), 2)
                cv2.putText(display_frame, 'Red', (pos[0] + 10, pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            for pos in blue_positions:
                cv2.circle(display_frame, pos, 10, (255, 0, 0), 2)
                cv2.putText(display_frame, 'Blue', (pos[0] + 10, pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            for pos in robot_positions:
                cv2.circle(display_frame, pos, 10, (0, 255, 0), 2)
                cv2.putText(display_frame, 'Robot', (pos[0] + 10, pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Affichage de la frame annotée
            cv2.imshow("Received Frame with Detections", display_frame)

        # Gestion de la touche pour quitter
        ch = chr(cv2.waitKey(1) & 0xFF)
        if ch == 'q' or ch == 'Q':
            stopProgram.set()

    # Fermeture propre
    stopThread.set()
    while threadRunning.is_set():
        pass
    sock.close()
    cv2.destroyAllWindows()


main()
