import pygame
import random
import math

# Get it to work with sigmoid, then work with ReLU and other activation functions

# self.weights.append(random.uniform(0, 1))


def mapRange(input, minInput, maxInput, minOutput, maxOutput):
    output = minOutput + ((maxOutput - minOutput) /
                          (maxInput - minInput)) * (input - minInput)
    return output


# Initialize pygame
pygame.init()

width = 600
height = 600
pygame.display.set_caption('Perceptron')
background_colour = (0, 0, 0)

# set the window
window = pygame.display.set_mode((width, height))
window.fill(background_colour)

pygame.display.flip()
running = True

mouse = pygame.mouse.get_pos()
left = False
middle = False
right = False

points = []

epochs = 10


def sigmoid(x):
    return 1/(1+math.exp(-x))


class Point:
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type
        self.color = (0, 255, 0)
        if (type == -1):
            self.color = (255, 255, 0)

    def render(self):
        pygame.draw.circle(window, self.color, (self.x, self.y), 5)


class Perceptron:
    def __init__(self):
        self.learning_rate = 0.1
        self.weights = [random.uniform(0, 1), random.uniform(0, 1)]
        self.bias = random.uniform(0, 1)

    def train(self, x0, x1, answer):
        output = sigmoid(x0*self.weights[0]+x1*self.weights[1]+self.bias)
        error = answer - output
        if (error < 0):
            error += 1
        self.weights[0] += error * x0 * self.learning_rate
        self.weights[1] += error * x1 * self.learning_rate
        self.bias += error * self.learning_rate

    def test(self, points):
        correct = 0;
        for point in points:
            x0 = mapRange(point.x, 0, width, 0, 1)
            x1 = mapRange(point.y, 0, height, 1, 0)
            output = sigmoid(x0*self.weights[0]+x1*self.weights[1]+self.bias)
            if (output < 0.5):
                 if (point.type == -1):
                     correct+=1
            else:
                if (point.type == 1):
                     correct+=1      
        return correct/len(points)

        
# Create the model
perceptron = Perceptron()

# animation loop
while running:

    pygame.event.get()
    # mouse list
    x, y = pygame.mouse.get_pos()
    # update screen
    pygame.display.flip()
    window.fill(background_colour)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_presses = pygame.mouse.get_pressed()
            left, middle, right = mouse_presses

            if left:
                p = Point(x, y, -1)
                points.append(p)
            elif right:
                p = Point(x, y, 1)
                points.append(p)
            elif middle:
                random.shuffle(points)
                accuracy = perceptron.test(points)
                print(f"Accuracy: {accuracy}")
                for i in range(epochs):
                    for point in points:
                        x0 = mapRange(point.x, 0, width, 0, 1)
                        x1 = mapRange(point.y, 0, height, 1, 0)
                        perceptron.train(x0, x1, point.type)
                accuracy = perceptron.test(points)
                print(f"Accuracy: {accuracy}")
                


    for point in points:
        point.render()
