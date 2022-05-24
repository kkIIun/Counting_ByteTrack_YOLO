def is_in_line(centroid, line, margin):
    print(centroid, line)
    if (centroid[1] > line[0][1] - margin) and (centroid[1] < line[0][1] + margin):
        return True
    return False
