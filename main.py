import pygame
import math
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def interpolate_points(points, num_points):
    points = np.array(points)

    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    total_distance = cumulative_distances[-1]
    normalized_distances = cumulative_distances / total_distance

    interp_func_x = interp1d(normalized_distances, points[:, 0], kind='linear')
    interp_func_y = interp1d(normalized_distances, points[:, 1], kind='linear')

    even_spaced = np.linspace(0, 1, num_points)
    new_points = [(int(interp_func_x(t)), int(interp_func_y(t))) for t in even_spaced]

    return new_points


def find_closest_point(point, points):
    min_distance = float('inf')
    closest_point = None
    for p in points:
        distance = math.dist(point, p)
        if distance < min_distance:
            min_distance = distance
            closest_point = p
    return closest_point

points = [
    [(780,207),(830,207)],
    [(780,267),(830,267)],
    [(780,307),(830,307)],
    [(780,367),(830,367)],
    [(780,417),(830,417)],
    [(780,457),(830,457)],
    [(780,507),(830,507)],
    [(780,567),(830,567)],
    [(1015,616),(1003,564)],
    [(835,588),(811,628)],
    [(875,633),(873,578)],
    [(936,567),(946,624)],
    [(1049,552),(1089,579)],
    [(1117,533),(1072,485)],
    [(1130,436),(1153,489)],
    [(1213,475),(1213,421)],
    [(1276,437),(1307,482)],
    [(1281,409),(1330,414)],
    [(1321,339),(1380,338)],
    [(1290,307),(1323,265)],
    [(1206,269),(1225,221)],
    [(1104,223),(1125,170)],
    [(1022,187),(1041,134)],
    [(924,148),(949,101)],
    [(843,113),(861,61)],
    [(762,81),(773,23)],
    [(726,66),(679,13)],
    [(684, 114), (668, 61)],
    [(592,124),(573,70)],
    [(519,147),(477,102)],
    [(454,189),(411,156)],
    [(391,242),(345,209)],
    [(337,287),(297,254)],
    [(245,296),(277,340)],
    [(195,337),(227,383)],
    [(155,378),(197,416)],
    [(165,458),(118,431)],
    [(85,470),(124,509)],
    [(84,562),(37,538)],
    [(8,599),(70,583)],
    [(68,637),(113,601)],
    [(121,633),(100,683)],
    [(153,692),(156,641)],
    [(197,644),(202,697)],
    [(257,651),(211,648)],
    [(263,604),(213,604)],
    [(263,554),(213,554)],
    [(263,504),(213,504)],
    [(263,454),(213,454)],
    [(273,423),(231,392)],
    [(306,397),(274,353)],
    [(338,360),(312,317)],
    [(377,353),(377,301)],
    [(419,350),(419,298)],
    [(433,351),(481,306)],
    [(431,388),(485,390)],
    [(459,449),(484,402)],
    [(518,458),(520,400)],
    [(557,403),(581,458)],
    [(618,409),(565,393)],
    [(618,350),(563,350)],
    [(618,300),(563,300)],
    [(618,250),(565,250)],
    [(618,200),(565,180)],
    [(633,142),(659,191)],
    [(695,178),(675,133)],
    [(745,163),(732,112)],
    [(779,157),(819,120)],
]
output_file = "reward_barriers.txt"
with open(output_file, "w") as file:
    for point in points:
        one = point[0]
        two = point[1]
        file.write(f"{one},{two}\n")

def draw_lines(points,line_color):
    for point in points:
        pygame.draw.line(screen, line_color, point[0], point[1], 2)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WALL_COLOR = (255, 0, 0)  # Color for the track (e.g., red)
ROAD_COLOR = (100, 100, 100)  # Color for the wider road (gray)

# Load the resized coordinates from the txt file
ins = "resized_track_coordinates.txt"
inside_wall = []
with open(ins, "r") as file:
    for line in file:
        x, y = map(int, line.strip().split(", "))
        inside_wall.append((x, y))
outs = "track_coordinates.txt"
outside_wall = []
with open(outs, "r") as file:
    for line in file:
        x, y = map(int, line.strip().split(", "))
        outside_wall.append((x, y))

class Wall:
    def __init__(self, x1, y1, x2, y2, color=(255, 0, 0), thickness=5):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.color = color
        self.thickness = thickness

    def draw(self, screen):
        # Draw the wall as a line between (x1, y1) and (x2, y2)
        pygame.draw.line(screen, self.color, (self.x1, self.y1), (self.x2, self.y2), self.thickness)

# Function to read the contour points from a file
def load_contour(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            x1, y1 = map(int, lines[i].strip().split(','))
            if i+1 >= len(lines):
                second = 0
            else:
                second = i+1
            x2, y2 = map(int, lines[second].strip().split(','))
            coordinates.append((x1, y1, x2, y2))
    return coordinates


walls_ins = [Wall(x1, y1, x2, y2) for x1, y1, x2, y2 in load_contour(ins)]
walls_out = [Wall(x1, y1, x2, y2) for x1, y1, x2, y2 in load_contour(outs)]

def distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def scale_point(x_out, y_out, ins_coords, scale_factor):
    closest_point = min(ins_coords, key=lambda p: distance(x_out, y_out, p[0], p[1]))
    x_in, y_in = closest_point

    dx = x_out - x_in
    dy = y_out - y_in

    x_scaled = x_in + dx * scale_factor
    y_scaled = y_in + dy * scale_factor

    return int(x_scaled), int(y_scaled)
def scale_outer_wall(walls_ins, walls_out, scale_factor=2):
    scaled_walls_out = []

    ins_coords = [(wall.x1, wall.y1) for wall in walls_ins] + [(wall.x2, wall.y2) for wall in walls_ins]

    for wall in walls_out:
        x1_out, y1_out, x2_out, y2_out = wall.x1, wall.y1, wall.x2, wall.y2

        new_x1_out, new_y1_out = scale_point(x1_out, y1_out, ins_coords, scale_factor)
        new_x2_out, new_y2_out = scale_point(x2_out, y2_out, ins_coords, scale_factor)

        scaled_walls_out.append(Wall(new_x1_out, new_y1_out, new_x2_out, new_y2_out))

    return scaled_walls_out

MARGIN = 10

WI = []
for j in range(len(walls_ins)):
    WI.append((walls_ins[j].x1,walls_ins[j].y1))
    WI.append((walls_ins[j].x2, walls_ins[j].y2))

WO = []
for j in range(len(walls_out)):
    WO.append([walls_out[j].x1,walls_out[j].y1])
    WO.append([walls_out[j].x2, walls_out[j].y2])

# Initialize Pygame
pygame.init()

# Set up the window dimensions
width, height = 1400, 700
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Racing")
# Main loop
running = True
font = pygame.font.Font(None, 36)  # Default font, size 36
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(WHITE)
    for wall in walls_ins:
        wall.draw(screen)
    pygame.draw.polygon(screen, ROAD_COLOR, WI)
    for wall1 in walls_out:
        wall1.draw(screen)
    pygame.draw.polygon(screen, WHITE, WO)
    draw_lines(points,(0, 0, 255))
    mouse_x, mouse_y = pygame.mouse.get_pos()
    text = font.render(f"Mouse: ({mouse_x}, {mouse_y})", True, (0, 0, 0))  # White text
    screen.blit(text, (10, 10))
    pygame.display.flip()

# Quit Pygame
pygame.quit()
