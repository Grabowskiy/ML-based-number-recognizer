import pygame
import numpy as np
import random
import sys

#If you want to use numpy based model:
from main import neur


#Global variables for grid and pixel management
CELL_SIZE = 12
GRID_SPACE = 1

#Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
OFF_WHITE = (245, 245, 245)
GRAY = (200, 200, 200)

#TODO Shading around a pixel for more realistic numbers
class Pixel():
    def __init__(self, id):
        self.id = id
        self.value = 0
        self.connections = 0

    def clicked(self, data, i, j):
        self.value = random.uniform(0.95, 1.0).double()
        check_connections(data, i, j)

    #Not in use yet: will be for shading the surrounding pixels
    def check_connections(self):
        if i > 0 and i < 27: a = True
        if j > 0 and j < 27: b = True
        # Chech lower, upper connection
        if a:
            if data[i-1][j].value > 0:
                self.connections += 1
            if data[i+1][j].value > 0:
                self.connections += 1
        # Check side connection
        if b:
            if data[i][j-1].value > 0:
                self.connections += 1
            if data[i][j+1].value > 0:
                self.connections += 1
        #Check diagonal connection
        if a and b:
            if data[i-1][j - 1].value > 0:
                self.connections += 1
            if data[i-1][j + 1].value > 0:
                self.connections += 1
            if data[i+1][j - 1].value > 0:
                self.connections += 1
            if data[i+1][j + 1].value > 0:
                self.connections += 1
        return self.connections

def create_data():
    #Create the numpy array
    array = []
    for i in range(784):
        new_pixel = Pixel(i)
        array.append(new_pixel)
    array = np.reshape(np.array(array), (1, 784))
    return array

def draw_grid(screen, grid):
    for row in range(28):
        for column in range(28):
            rect = pygame.Rect(column * (CELL_SIZE + GRID_SPACE) + GRID_SPACE, row * (CELL_SIZE + GRID_SPACE) + GRID_SPACE,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, grid[row][column], rect)

def draw_button(screen, position: tuple, height, title, t_size):
    font = pygame.font.Font(None, t_size)
    rect = pygame.Rect(position[0], position[1], 200, height)
    pygame.draw.rect(screen, OFF_WHITE, rect, border_radius=10)
    title_surface = font.render(title, True, BLACK)
    title_rect = title_surface.get_rect(center=rect.center)
    screen.blit(title_surface, title_rect)
    return rect

#Main window
def consol():
    mouse_down = False
    #Make the pixels
    data = create_data()
    #grid state
    grid = [[WHITE for _ in range(28)] for _ in range(28)]
    #Initialise pygame
    pygame.init()
    screen = pygame.display.set_mode((700, 385))
    pygame.display.set_caption('ML number recognition')
    clock = pygame.time.Clock()

    #Running loop
    running = True
    while running:
        draw_grid(screen, grid)
        go_button = draw_button(screen, (440, 30),50, 'GO', 36)
        clear_button = draw_button(screen, (440, 285), 50, 'CLEAR', 36)

        #Event handlings
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            #Checking if any buttin is clicked
            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_x, mouse_y = event.pos
                #'GO' button
                if go_button.collidepoint(mouse_x, mouse_y):
                    data_values = [data[0][x].value for x in range(784)]
                    no_input = all(value == 0 for value in data_values)
                    if not no_input:
                        data_values = np.reshape(np.array(data_values), (1, 784))
                        prediction = neur.forvard(data_values)
                        prediction = np.argmax(prediction)
                        answer = draw_button(screen, (440, 120),100,f"{prediction}", 56)
                #'CLEAR' button
                if clear_button.collidepoint(mouse_x, mouse_y):
                    data = create_data()
                    grid = [[WHITE for _ in range(28)] for _ in range(28)]
                mouse_down = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True
            elif event.type == pygame.MOUSEMOTION and mouse_down:
                mouse_x, mouse_y = event.pos
                # Calculate the grid position
                grid_x = mouse_x // (CELL_SIZE + GRID_SPACE)
                grid_y = mouse_y // (CELL_SIZE + GRID_SPACE)
                if 0 <= grid_x < 28 and 0 <= grid_y < 28:
                    # Calculate the exact rectangle to change color
                    rect = pygame.Rect(
                        grid_x * (CELL_SIZE + GRID_SPACE) + GRID_SPACE,
                        grid_y * (CELL_SIZE + GRID_SPACE) + GRID_SPACE,
                        CELL_SIZE,
                        CELL_SIZE
                    )
                    grid[grid_y][grid_x] = BLACK

                    pixel_id = grid_y * 28 + grid_x
                    data[0][pixel_id].value = random.uniform(0.8500, 0.99999)
                    #TODO jelenleg egy tuple ként adja ki melyik grid re nyomtál rá, ebből ki lehet szedni melyik pixelnek
                    # kell feketének lennie és át lehet nyomni a value ját, ezt a colort kell valahogy átadni a rajzolásnak

        pygame.display.flip()
    pygame.quit()
    sys.exit()

#Starting the consol
consol()