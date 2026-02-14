LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_region(landmarks, eye_indices, width, height):
    points = []
    
    for index in eye_indices:
        point = landmarks[index]
        x = int(point.x * width)
        y = int(point.y * height)
        points.append((x, y))
        
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    return x1, y1, x2, y2