def create_rectangle(coords, width, height):
    min_x = min([point.x for point in coords])
    max_x = max([point.x for point in coords])
    min_y = min([point.y for point in coords])
    max_y = max([point.y for point in coords])
    
    x1 = int(min_x * width)
    y1 = int(min_y * height)
    
    x2 = int(max_x * width)
    y2 = int(max_y * height)

    return (x1, y1, x2, y2)