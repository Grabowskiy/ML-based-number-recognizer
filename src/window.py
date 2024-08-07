import main
import pygame
import numpy as np
import random
import sys
#Using numpy based model
import main


#Screen
HEIGHT = 385
WIDTH = 700

#Global variables for grid and pixel management
CELL_SIZE = 12
GRID_SPACE = 0

#Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
OFF_WHITE = (245, 245, 245)
GRAY = (200, 200, 200)

#Font
font_path = "data/ProductSans-Regular.ttf"

class Pixel():
    def __init__(self, id):
        self.id = id
        self.value = 0.0

    def clicked(self, data, grid, grid_x, grid_y):
        self.value = random.uniform(0.90, 1.0)
        grid[grid_y][grid_x] = BLACK

        for i in range(grid_y-1, grid_y+2):
            for j in range(grid_x-1, grid_x+2):
                try:
                    pixel_id = i * 28 + j
                    pixel = data[0][pixel_id]
                except IndexError:
                    continue

                if pixel.value == 0.0:
                    pixel.value = random.uniform(0.4, 0.6)
                if i != grid_y or j != grid_x:
                    self.add_color_to_neighbour(pixel, grid, pixel_id)

    def add_color_to_neighbour(self, pixel, grid, pixel_id):
        #So the sides don't spill to the other side
        if pixel.value > 0.89 or pixel_id % 28 == 0 or pixel_id % 28 == 27 \
                and self.id - 28 != pixel_id and self.id + 28 != pixel_id:
            return

        graycolor_value = GRAY[0] * pixel.value
        color = graycolor_value, graycolor_value, graycolor_value
        row = int(pixel_id / 28)
        column = int(pixel_id % 28)
        grid[row][column] = color


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
    font = pygame.font.Font(font_path, t_size)
    rect = pygame.Rect(position[0], position[1], 200, height)
    pygame.draw.rect(screen, OFF_WHITE, rect, border_radius=10)
    title_surface = font.render(title, True, BLACK)
    title_rect = title_surface.get_rect(center=rect.center)
    screen.blit(title_surface, title_rect)
    return rect

#Main window
def consol(neur: main.NeuralNetwork):
    mouse_down = False
    #Make the pixels
    data = create_data()
    #grid state
    grid = [[WHITE for _ in range(28)] for _ in range(28)]
    #Initialise pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('ML number recognition')
    clock = pygame.time.Clock()

    text_size = 24
    text_font = pygame.font.Font(font_path, text_size)
    text = "Draw a number from 0 to 9"
    text_surface = text_font.render(text, True, WHITE)

    #Running loop
    running = True
    while running:
        draw_grid(screen, grid)
        go_button = draw_button(screen, (440, 30),50, 'GO', 36)
        clear_button = draw_button(screen, (440, 285), 50, 'CLEAR', 36)

        text_rect = text_surface.get_rect(center=(CELL_SIZE * 14, CELL_SIZE * 28 + (HEIGHT - CELL_SIZE * 28) / 2))
        screen.blit(text_surface, text_rect)

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
                    pixel_id = grid_y * 28 + grid_x
                    pixel = data[0][pixel_id]
                    pixel.clicked(data, grid, grid_x, grid_y)

        pygame.display.flip()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    #Start learning
    neur = main.main()

    #Starting the consol
    consol(neur)