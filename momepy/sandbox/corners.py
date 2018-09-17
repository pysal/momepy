import numpy as np
import statistics


a = np.array([32.49, -39.96])
b = np.array([31.39, -39.28])
c = np.array([31.14, -38.09])

ba = a - b
bc = c - b

cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
angle = np.arccos(cosine_angle)

print(np.degrees(angle))


def angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))

    return angle


points = [(0.0, 0.0), (2.0, 1.0), (2.0, 2.0), (4.0, 6.0), (4.0, 0.0), (0.0, 0.0)]
angles = []
stop = len(points) - 1  # define where to stop
for i in np.arange(len(points)):  # for every point, calculate angle and add 1 if True angle
    if i == 0:
            continue
    elif i == stop:
        a = np.asarray(points[i - 1])
        b = np.asarray(points[i])
        c = np.asarray(points[1])
        ang = angle(a, b, c)

        if ang <= 170:
            angles.append(ang)
        elif angle(a, b, c) >= 190:
            angles.append(ang)
        else:
            continue

    else:
        a = np.asarray(points[i - 1])
        b = np.asarray(points[i])
        c = np.asarray(points[i + 1])
        ang = angle(a, b, c)

        if angle(a, b, c) <= 170:
            angles.append(ang)
        elif angle(a, b, c) >= 190:
            angles.append(ang)
        else:
            continue

angles

deviations = []
for i in angles:
    dev = abs(90 - i)
    deviations.append(dev)

np.mean(deviations)

corners = 0

stop = len(points) - 1

for i in np.arange(len(points)):
    if i == 0:
        print('do A', i)
        a = np.asarray(points[-1])
        b = np.asarray(points[i])
        c = np.asarray(points[i + 1])

        if angle(a, b, c) is True:
            corners = corners + 1
            print('A', i)
        else:
            print('A go', i)
            continue
    elif i == stop:
        print('do B', i)
        a = np.asarray(points[i - 1])
        b = np.asarray(points[i])
        c = np.asarray(points[0])

        if angle(a, b, c) is True:
            corners = corners + 1
            print('B', i)
        else:
            print('B go', i)
            continue

    else:
        print('do C', i)
        a = np.asarray(points[i - 1])
        b = np.asarray(points[i])
        c = np.asarray(points[i + 1])

        if angle(a, b, c) is True:
            corners = corners + 1
            print('C', i)
        else:
            print('C go', i)
            continue

corners

i = 1
a = np.asarray(points[i - 1])
b = np.asarray(points[i])
c = np.asarray(points[i + 1])
a
b
c
if angle(a, b, c) is True:
    corners = corners + 1


points = [(0.0, 0.0), (2.0, 1.0), (2.0, 2.0), (4.0, 6.0), (4.0, 0.0), (0.0, 0.0)]


def true_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    if np.degrees(angle) <= 170:
        return True
    elif np.degrees(angle) >= 190:
        return True
    else:
        return False

# fill new column with the value of area, iterating over rows one by one
for i in np.arange(len(points)):
    distances = []
    centroid = row['geometry'].centroid
    points = list(row['geometry'].exterior.coords)  # get points of a shape
    stop = len(points) - 1  # define where to stop
    for i in np.arange(len(points)):  # for every point, calculate angle and add 1 if True angle
        if i == 0:
                continue
        elif i == stop:
            a = np.asarray(points[i - 1])
            b = np.asarray(points[i])
            c = np.asarray(points[1])

            if true_angle(a, b, c) is True:
                distance = centroid.distance(i)
                distances.append(distance)
            else:
                continue

        else:
            a = np.asarray(points[i - 1])
            b = np.asarray(points[i])
            c = np.asarray(points[i + 1])

            if true_angle(a, b, c) is True:
                distance = centroid.distance(i)
                distances.append(distance)
            else:
                continue

    objects.loc[index, column_name] = np.mean(distances)
