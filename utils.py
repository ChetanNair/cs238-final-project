import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        
class Line:
    def __init__(self, pt1, pt2):
        self.pt1 = Point(pt1.x, pt1.y)
        self.pt2 = Point(pt2.x, pt2.y)
    
    @property
    def x1(self):
        return self.pt1.x

    @property
    def y1(self):
        return self.pt1.y

    @property
    def x2(self):
        return self.pt2.x

    @property
    def y2(self):
        return self.pt2.y
        
class Ray:
    def __init__(self, position, angle):
        self.position = position
        self.angle = angle

    def cast(self, wall):
        x1, y1, x2, y2 = wall.x1, wall.y1, wall.x2, wall.y2
        vec = rotate(Point(0, 0), Point(0, -1000), self.angle)
        x3, y3 = self.position.x, self.position.y
        x4, y4 = self.position.x + vec.x, self.position.y + vec.y
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if denominator == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        if 0 < t < 1 and 0 < u < 1:
            intersect_x = math.floor(x1 + t * (x2 - x1))
            intersect_y = math.floor(y1 + t * (y2 - y1))
            return Point(intersect_x, intersect_y)
        
        return None

def distance(pt1, pt2):
    return(((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)**0.5)

def rotate(origin, point, angle):
    qx = origin.x + math.cos(angle) * (point.x - origin.x) - math.sin(angle) * (point.y - origin.y)
    qy = origin.y + math.sin(angle) * (point.x - origin.x) + math.cos(angle) * (point.y - origin.y)
    q = Point(qx, qy)
    return q

def rotate_rect(pt1, pt2, pt3, pt4, angle):
    pt_center = Point((pt1.x + pt3.x)/2, (pt1.y + pt3.y)/2)

    pt1 = rotate(pt_center,pt1,angle)
    pt2 = rotate(pt_center,pt2,angle)
    pt3 = rotate(pt_center,pt3,angle)
    pt4 = rotate(pt_center,pt4,angle)

    return pt1, pt2, pt3, pt4

def line_intersection(line1, line2):
    x1, y1 = line1.pt1.x, line1.pt1.y
    x2, y2 = line1.pt2.x, line1.pt2.y
    x3, y3 = line2.pt1.x, line2.pt1.y
    x4, y4 = line2.pt2.x, line2.pt2.y

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return False

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

    return 0 <= t <= 1 and 0 <= u <= 1