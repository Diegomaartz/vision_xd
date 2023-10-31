import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

centers = []
colors = ["red","blue", "green", "magenta", "cyan", "yellow"]
c_i = 0

def random_points(center_x, center_y, num_points, radius):
    global c_i
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    distances = np.random.uniform(0, radius, num_points)
    x_coordinates = center_x + distances * np.cos(angles)
    y_coordinates = center_y + distances * np.sin(angles)
    return x_coordinates, y_coordinates

def onclick(event):
    global c_i
    center_x, center_y = event.xdata, event.ydata
    centers.append((center_x, center_y))

    num_random_points = 50
    radius = 20

    x_coordinates, y_coordinates = random_points(center_x, center_y, num_random_points, radius)

    ax.scatter(x_coordinates, y_coordinates, s=5, c=(colors[c_i]))
    plt.draw()
    c_i = (c_i + 1) % len(colors)


folder_path = 'bd_tuercas'
file_list = os.listdir(folder_path)
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
for file_name in file_list:
    if any(file_name.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(folder_path, file_name)
        img = plt.imread(image_path)

        fig, ax = plt.subplots()
        plt.title(file_name)
        ax.imshow(img)

        fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show()
plt.close('all')