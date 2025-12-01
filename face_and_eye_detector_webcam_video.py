import cv2
import pygame
import time
from twilio.rest import Client

# ========= SETTINGS =========
ALARM_FILE = "audio/alert.wav"          # make sure this file exists
EYES_CLOSED_FRAMES_THRESHOLD = 5       # lower = faster alarm

# Twilio settings – FILL THESE
ACCOUNT_SID = "AC10d94c6aca5001a528183be2f4b39933"
AUTH_TOKEN = "f0b7db686411299758e8b216e2059fa6"
TWILIO_NUMBER = "15856343217"         # your Twilio phone number
OWNER_NUMBER = "+919392777619"         # owner's mobile number
# ============================

# init pygame for audio
pygame.mixer.init()

# init Twilio client
twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)
last_call_time = 0     # to avoid calling too often
CALL_COOLDOWN_SECONDS = 60  # at most one call per 60 seconds


def call_owner_if_needed():
    """
    Place a voice call to the owner if cooldown has passed.
    """
    global last_call_time

    now = time.time()
    if now - last_call_time < CALL_COOLDOWN_SECONDS:
        # too soon since last call – skip
        return

    try:
        print("Placing Twilio call to owner...")
        call = twilio_client.calls.create(
            twiml='<Response><Say>Alert. The driver appears drowsy. '
                  'Please check immediately.</Say></Response>',
            to=OWNER_NUMBER,
            from_=TWILIO_NUMBER,
        )
        print("Call initiated. Sid:", call.sid)
        last_call_time = now
    except Exception as e:
        print("Error while making Twilio call:", e)


# Haarcascade models
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

# open webcam
video_capture = cv2.VideoCapture(0)

closed_frames = 0
alarm_on = False

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eyes_detected = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        for (ex, ey, ew, eh) in eyes:
            eyes_detected = True
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # ----- drowsiness logic -----
    if len(faces) > 0:  # process only when face is detected
        if eyes_detected:
            closed_frames = 0
            if alarm_on:
                pygame.mixer.music.stop()
                alarm_on = False
        else:
            closed_frames += 1
            print("Eyes closed frames:", closed_frames)
            if closed_frames >= EYES_CLOSED_FRAMES_THRESHOLD and not alarm_on:
                # play local alarm
                try:
                    pygame.mixer.music.load(ALARM_FILE)
                    pygame.mixer.music.play(-1)  # loop
                    alarm_on = True
                except Exception as e:
                    print("Error playing sound:", e)

                # CALL OWNER via Twilio
                call_owner_if_needed()

    if alarm_on:
        cv2.putText(
            frame,
            "DROWSY! WAKE UP!",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

    cv2.imshow("Drowsiness Detector", frame)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
