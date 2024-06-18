import cv2
import matplotlib.pyplot as plt

masks_points = [{'green': [], 'red': []}]
current_mask_index = 0
current_color = 'green'

def select_points(image_path):
    global img, img_copy, current_mask_index, current_color, masks_points
    img = cv2.imread(image_path)

    if img is None:
        print(f"Errore: Impossibile caricare l'immagine dal percorso '{image_path}'")
        return []

    img_copy = img.copy()
    cv2.imshow('Image', img_copy)
    cv2.setMouseCallback('Image', on_mouse_points)
    print("Premi 'g' per selezionare punti verdi, 'r' per punti rossi.")
    print("Premi 'q' per terminare, 'n' per nuova maschera, 'b' per tornare indietro.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            current_color = 'green'
            print("Modalità: punti verdi")
        elif key == ord('r'):
            current_color = 'red'
            print("Modalità: punti rossi")
        elif key == ord('n'):
            current_mask_index += 1
            if current_mask_index >= len(masks_points):
                masks_points.append({'green': [], 'red': []})
            update_image_points()
            print(f"Maschera {current_mask_index}")
        elif key == ord('b'):
            if current_mask_index > 0:
                current_mask_index -= 1
                update_image_points()
                print(f"Maschera {current_mask_index}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    return masks_points

def on_mouse_points(event, x, y, flags, param):
    global current_color, masks_points, current_mask_index
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_color == 'green':
            masks_points[current_mask_index]['green'].append((x, y))
            cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
        elif current_color == 'red':
            masks_points[current_mask_index]['red'].append((x, y))
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', img_copy)

def update_image_points():
    global img, img_copy, masks_points, current_mask_index
    img_copy = img.copy()
    for point in masks_points[current_mask_index]['green']:
        cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
    for point in masks_points[current_mask_index]['red']:
        cv2.circle(img_copy, point, 5, (0, 0, 255), -1)
    cv2.imshow('Image', img_copy)

def display_points(masks):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)

    for mask in masks:
        x_coords_green = [p[0] for p in mask['green']]
        y_coords_green = [p[1] for p in mask['green']]
        plt.scatter(x_coords_green, y_coords_green, color='green', s=10)

        x_coords_red = [p[0] for p in mask['red']]
        y_coords_red = [p[1] for p in mask['red']]
        plt.scatter(x_coords_red, y_coords_red, color='red', s=10)

    plt.show()


masks_boxes = [[]]
drawing = False
start_point = (0, 0)

def select_boxes(image_path):
    global img, img_copy, current_mask_index, masks_boxes, start_point, drawing
    img = cv2.imread(image_path)
    start_point = (0, 0)
    drawing = False

    if img is None:
        print(f"Errore: Impossibile caricare l'immagine dal percorso '{image_path}'")
        return []

    img_copy = img.copy()
    cv2.imshow('Image', img_copy)
    cv2.setMouseCallback('Image', on_mouse_boxes)
    print("Premi 'q' per terminare, 'n' per nuova maschera.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    return masks_boxes

def on_mouse_boxes(event, x, y, flags, param):
    global current_mask_index, masks_boxes, start_point, drawing
    if event == cv2.EVENT_LBUTTONDOWN and not drawing:
        start_point = (x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_temp = img.copy()
        for box in masks_boxes[current_mask_index]:
            cv2.rectangle(img_temp, box[0], box[1], (255, 0, 0), 2)
        cv2.rectangle(img_temp, start_point, (x, y), (255, 0, 0), 2)
        cv2.imshow('Image', img_temp)
    elif event == cv2.EVENT_LBUTTONDOWN and drawing:
        end_point = (x, y)
        masks_boxes[current_mask_index].append((start_point, end_point))
        drawing = False
        cv2.rectangle(img_copy, start_point, end_point, (255, 0, 0), 2)
        cv2.imshow('Image', img_copy)

def display_boxes(masks):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)

    for mask in masks:
        for box in mask:
            (start_x, start_y), (end_x, end_y) = box
            rect = plt.Rectangle((start_x, start_y), end_x - start_x, end_y - start_y,
                                 linewidth=2, edgecolor='blue', facecolor='none')
            plt.gca().add_patch(rect)

    plt.show()


def main(image_path):
    print("Seleziona la modalità: 'p' per punti, 'b' per bounding boxes.")
    key = input("Inserisci la tua scelta: ").strip().lower()
    if key == 'p':
        masks = select_points(image_path)
        display_points(masks)
    elif key == 'b':
        masks = select_boxes(image_path)
        display_boxes(masks)
    else:
        print("Scelta non valida")


main('./dataset2/images/0.png')