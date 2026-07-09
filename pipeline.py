import cv2
import numpy as np

ID_DA_CAMERA = 1
COR_INFERIOR = np.array([10, 50, 50])
COR_SUPERIOR = np.array([30, 255, 255])


AREA_MINIMA_BISCOITO_INTEIRO = 20000

cap = cv2.VideoCapture(ID_DA_CAMERA)


# def inspecionar_biscoito(roi):
#     """Analisa o interior do biscoito buscando quebras."""
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     # Retorna a quantidade de bordas brancas encontradas
#     return np.sum(edges)


def inspecionar_biscoito(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Threshold inverte a imagem: o biscoito fica escuro e os furos ficam claros
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    contornos_internos, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filtra contornos que são pequenos demais (ruído)
    num_furos = 0
    for c in contornos_internos:
        area = cv2.contourArea(c)
        if area > 50:
            num_furos += 1

    return num_furos


while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, COR_INFERIOR, COR_SUPERIOR)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        if area < 5000:
            continue

        if area < AREA_MINIMA_BISCOITO_INTEIRO * 0.7:
            color = (0, 165, 255)  # Laranja
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"Defeituosa (Area: {int(area)})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            continue

        roi = frame[y : y + h, x : x + w]
        score_quebra = inspecionar_biscoito(roi)

        if score_quebra > 150000:
            status = "Quebrado"
            color = (0, 0, 255)
        else:
            status = "Perfeito"
            color = (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{status} (Bordas: {int(score_quebra)})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Inspecao Industrial - Multibiscoito", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
