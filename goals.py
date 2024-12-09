import pygame

class Goal:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.isactiv = False
    
    def draw(self, win):
        pygame.draw.line(win, (0, 255, 0), (self.x1, self.y1), (self.x2, self.y2), 2)
        if self.isactiv:
            pygame.draw.line(win, (255, 0, 0), (self.x1, self.y1), (self.x2, self.y2), 2)

def get_goals():
    goals = []
    
    with open("reward_barriers.txt", "r") as file:
        lines = file.readlines()
    
    for line in lines:
        line = line.strip()
        if line:
            try:
                coord1, coord2 = line.split("),(")
                x1, y1 = map(int, coord1.strip("()").split(","))
                x2, y2 = map(int, coord2.strip("()").split(","))
                goals.append(Goal(x1, y1, x2, y2))
            except ValueError:
                print(f"Error parsing line: {line}")
    
    goals = goals[::-1]
    if goals:
        goals[-1].isactiv = True

    return goals