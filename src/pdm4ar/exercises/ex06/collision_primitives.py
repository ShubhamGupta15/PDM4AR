from pdm4ar.exercises_def.ex06.structures import *
from triangle import triangulate
import numpy as np


def distance(a:Point, b:Point) -> float:
    return np.linalg.norm(diff(a,b), 2)

def diff(a: Point, b: Point):
    return [a.x-b.x, a.y-b.y]

def area(t: Triangle):
    a = distance(t.v1,t.v2)
    b = distance(t.v2,t.v3)
    c = distance(t.v3,t.v1)
    s = (a+b+c)/2

    return np.sqrt(s*(s-a)*(s-b)*(s-c))


def seg_to_line(seg:Segment):
    x1 = seg.p1.x
    y1 = seg.p1.y
    x2 = seg.p2.x
    y2 = seg.p2.y

    a = y1-y2
    b = x2-x1
    c = x1*(y1-y2) + y1*(x2-x1)

    return [a,b,c]

def intersection_pt(p1,p2):
    [a1,b1,c1] = p1
    [a2,b2,c2] = p2

    A = np.array([[a1,b1],[a2,b2]])
    b= np.array([[c1,c2]]).T
    try:
        x = np.linalg.solve(A,b)
    except:
        if a1/a2 == c1/c2:
            return [np.inf]
        else:
            return []

    
    return [Point(x[0],x[1])]


def ccw(A,B,C):
    return (C.y-A.y)*(B.x-A.x) > (B.y-A.y)*(C.x-A.x)

class CollisionPrimitives:
    """
    Class of collision primitives
    """

    @staticmethod
    def point_line_collison(seg: Segment, p: Point) -> bool:

        d1 = distance(seg.p1, p)
        d2 = distance(seg.p2,p)
        leng = distance(seg.p1,seg.p2)
        buffer = 1e-9

        if d1+d2 <= leng+buffer:
            return True
        else:
            return False
        

    def segemnt_segment_collision(s1: Segment, s2: Segment):
        A = s1.p1
        B = s1.p2
        C = s2.p1
        D = s2.p2
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


    @staticmethod
    def circle_point_collision(c: Circle, p: Point) -> bool:
        if distance(c.center, p) > c.radius:
            return False
        else:
            return True 

    @staticmethod
    def triangle_point_collision(t: Triangle, p: Point) -> bool:
        area_orig = area(t)
        t1 = Triangle(t.v1,t.v2,p)
        at1 = area(t1)
        t2 = Triangle(t.v1,p,t.v3)
        at2 = area(t2)
        t3 = Triangle(p,t.v2,t.v3)
        at3 = area(t3)

        margin = 1e-6

        if area_orig-(at1+at2+at3)> -margin and area_orig-(at1+at2+at3) < margin:
            return True
        
        return False


    @staticmethod
    def polygon_point_collision(poly: Polygon, p: Point) -> bool:
        vertices = []
        for vertex in poly.vertices:
            vertices.append([vertex.x, vertex.y])

        t = triangulate({'vertices':vertices}, 'D')

        tri_vec = t['vertices'].tolist()
        tri = t['triangles'].tolist()

        for triangle in tri:
            v1 = tri_vec[triangle[0]]
            v2 = tri_vec[triangle[1]]
            v3 = tri_vec[triangle[2]]
            
            v1 = Point(v1[0],v1[1])
            v2 = Point(v2[0],v2[1])
            v3 = Point(v3[0],v3[1])
            trian = Triangle(v1,v2,v3)

            if CollisionPrimitives.triangle_point_collision(trian, p):
                return True
        return False

    @staticmethod
    def circle_segment_collision(c: Circle, segment: Segment) -> bool:
        if CollisionPrimitives.circle_point_collision(c,segment.p1) or CollisionPrimitives.circle_point_collision(c,segment.p2):
            return True
        
        center = np.array([c.center.x,c.center.y])
        p1 = np.array([segment.p1.x, segment.p1.y])
        p2 = np.array([segment.p2.x, segment.p2.y])
        
        dot = np.dot(center-p1, p2-p1)/distance(segment.p1, segment.p2)

        closest = p1 + (dot*(p2-p1))/distance(segment.p1, segment.p2)
        closest_point = Point(closest[0], closest[1])

        if not CollisionPrimitives.point_line_collison(segment, closest_point):
            return False
        
        dist = np.linalg.norm(center-closest)
        if dist > c.radius:
            return False
        else:
            return True


    @staticmethod
    def triangle_segment_collision(t: Triangle, segment: Segment) -> bool:
        try:
            if CollisionPrimitives.triangle_point_collision(t,segment.p1):
                return True
            if CollisionPrimitives.triangle_point_collision(t,segment.p2):
                return True
        except:
            print(segment)

        s1 = Segment(t.v1,t.v2)
        l1 = seg_to_line(s1)

        s2 = Segment(t.v1,t.v3)
        l2 = seg_to_line(s2)

        s3 = Segment(t.v2,t.v3)
        l3 = seg_to_line(s3)


        return CollisionPrimitives.segemnt_segment_collision(s1,segment) or CollisionPrimitives.segemnt_segment_collision(s2,segment) or CollisionPrimitives.segemnt_segment_collision(s3,segment)


    @staticmethod
    def polygon_segment_collision(p: Polygon, segment: Segment) -> bool:
        vertices = []
        for vertex in p.vertices:
            vertices.append([vertex.x, vertex.y])

        t = triangulate({'vertices':vertices})

        tri_vec = t['vertices'].tolist()
        tri = t['triangles'].tolist()

        for triangle in tri:
            v1 = tri_vec[triangle[0]]
            v2 = tri_vec[triangle[1]]
            v3 = tri_vec[triangle[2]]
            
            v1 = Point(v1[0],v1[1])
            v2 = Point(v2[0],v2[1])
            v3 = Point(v3[0],v3[1])
            trian = Triangle(v1,v2,v3)

            if CollisionPrimitives.triangle_segment_collision(trian, segment):
                return True
        return False


    @staticmethod
    def polygon_segment_collision_aabb(p: Polygon, segment: Segment) -> bool:

        x_min = np.inf
        y_min = np.inf
        x_max = -np.inf
        y_max = -np.inf
        for vertex in p.vertices:
            x_min = min(vertex.x, x_min)
            x_max = max(vertex.x, x_max)
            y_min = min(vertex.y, y_min)
            y_max = max(vertex.y, y_max)
            
        if segment.p1.x < x_min and segment.p2.x < x_min:
            return False
        elif segment.p1.y < y_min and segment.p2.y < y_min:
            return False
        elif segment.p1.x > x_max and segment.p2.x > x_max:
            return False
        elif segment.p1.y > y_max and segment.p2.y > y_max:
            return False
        else:
            return CollisionPrimitives.polygon_segment_collision(p,segment)

    @staticmethod
    def _poly_to_aabb(g: Polygon) -> AABB:
        # todo feel free to implement functions that upper-bound a shape with an
        #  AABB or simpler shapes for faster collision checks
        return AABB(p_min=Point(0, 0), p_max=Point(1, 1))
    


    @staticmethod
    def circle_circle_collision(c1:Circle,c2:Circle):
        if distance(c1.center, c2.center) > c1.radius + c2.radius:
            return False
        else:
            return True

    @staticmethod
    def circle_triangle_collision(c: Circle, t: Triangle):

        if CollisionPrimitives.triangle_point_collision(t,c.center):
            return True
        if CollisionPrimitives.circle_point_collision(c,t.v1):
            return True
        if CollisionPrimitives.circle_point_collision(c,t.v2):
            return True
        if CollisionPrimitives.circle_point_collision(c,t.v3):
            return True

        seg = [Segment(t.v1,t.v2), Segment(t.v2,t.v3), Segment(t.v3,t.v1)]
        for s in seg:
            if CollisionPrimitives.circle_segment_collision(c,s):
                return True
            
        return False

    @staticmethod
    def circle_polygon_collision(c: Circle, p:Polygon):

        if CollisionPrimitives.polygon_point_collision(p,c.center):
            return True
        
        for vertex in p.vertices:
            if CollisionPrimitives.circle_point_collision(c,vertex):
                return True

        for i in range(len(p.vertices)-1):
            seg = Segment(p.vertices[i], p.vertices[i+1])
            if CollisionPrimitives.circle_segment_collision(c, seg):
                return True

        return False