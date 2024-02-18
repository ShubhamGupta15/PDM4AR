from typing import List, Tuple
from dg_commons import SE2Transform
from pdm4ar.exercises.ex06.collision_primitives import CollisionPrimitives
from pdm4ar.exercises_def.ex06.structures import (
    Polygon,
    GeoPrimitive,
    Point,
    Segment,
    Circle,
    Triangle,
    Path,
)
import shapely
import numpy as np

##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not

COLLISION_PRIMITIVES = {
    Point: {
        Circle: lambda x, y: CollisionPrimitives.circle_point_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_point_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_point_collision(y, x),
    },
    Segment: {
        Circle: lambda x, y: CollisionPrimitives.circle_segment_collision(y, x),
        Triangle: lambda x, y: CollisionPrimitives.triangle_segment_collision(y, x),
        Polygon: lambda x, y: CollisionPrimitives.polygon_segment_collision_aabb(y, x),
    },
    Triangle: {
        Point: CollisionPrimitives.triangle_point_collision,
        Segment: CollisionPrimitives.triangle_segment_collision,
        Circle: lambda x,y : CollisionPrimitives.circle_triangle_collision(y,x),
    },
    Circle: {
        Point: CollisionPrimitives.circle_point_collision,
        Segment: CollisionPrimitives.circle_segment_collision,
        Circle: CollisionPrimitives.circle_circle_collision,
        Triangle: CollisionPrimitives.circle_triangle_collision,
        Polygon: CollisionPrimitives.circle_polygon_collision,
    },
    Polygon: {
        Point: CollisionPrimitives.polygon_point_collision,
        Segment: CollisionPrimitives.polygon_segment_collision_aabb,
        Circle: lambda x,y : CollisionPrimitives.circle_polygon_collision(y,x),
    },
}


def check_collision(p_1: GeoPrimitive, p_2: GeoPrimitive) -> bool:
    """
    Checks collision between 2 geometric primitives
    Note that this function only uses the functions that you implemented in CollisionPrimitives class.
        Parameters:
                p_1 (GeoPrimitive): Geometric Primitive
                p_w (GeoPrimitive): Geometric Primitive
    """
    assert type(p_1) in COLLISION_PRIMITIVES, "Collision primitive does not exist."
    assert (
        type(p_2) in COLLISION_PRIMITIVES[type(p_1)]
    ), "Collision primitive does not exist."

    collision_func = COLLISION_PRIMITIVES[type(p_1)][type(p_2)]

    return collision_func(p_1, p_2)


##############################################################################################
############################# This is a helper function. #####################################
# Feel free to use this function or not

def geo_primitive_to_shapely(p: GeoPrimitive):
    if isinstance(p, Point):
        return shapely.Point(p.x, p.y)
    elif isinstance(p, Segment):
        return shapely.LineString([[p.p1.x, p.p1.y], [p.p2.x, p.p2.y]])
    elif isinstance(p, Circle):
        return shapely.Point(p.center.x, p.center.y).buffer(p.radius)
    elif isinstance(p, Triangle):
        return shapely.Polygon([[p.v1.x, p.v1.y], [p.v2.x, p.v2.y], [p.v3.x, p.v3.y]])
    else: #Polygon
        vertices = []
        for vertex in p.vertices:
            vertices += [(vertex.x, vertex.y)]
        return shapely.Polygon(vertices)



class Grid():
    def __init__(self, res_x:float,
                    res_y:float,
                    origin: Point,
                    occ_grid_cells_x:int,
                    occ_grid_cells_y:int,
                    obstacles: List[GeoPrimitive]) -> None:
        
        self.grid = np.zeros([occ_grid_cells_x, occ_grid_cells_y])
        self.res_x = res_x
        self.res_y =res_y
        self.origin = origin
        self.occ_x = occ_grid_cells_x
        self.occ_y = occ_grid_cells_y
        
        for obstacle in obstacles:
            self._add_obstacle(obstacle)

    def _add_obstacle(self, obstacle:GeoPrimitive):
        polygon = geo_primitive_to_shapely(obstacle)

        [min_x, min_y, max_x, max_y] = shapely.bounds(polygon).tolist()
        for x in np.arange(min_x,max_x , self.res_x):
            for y in np.arange(min_y, max_y , self.res_y):
                point_shapely = geo_primitive_to_shapely(Point(x, y))
                if polygon.contains(point_shapely):
                    [i,j] = self._world_to_grid(Point(x, y))
                    self.grid[i,j] = 1
    
    def _world_to_grid(self,world_point: Point):
        grid_x = int((world_point.x - self.origin.x) / self.res_x)
        grid_y = int((world_point.y - self.origin.y) / self.res_y)
        grid_x = min(grid_x, self.occ_x-1)
        grid_y = min(grid_y, self.occ_y-1)
        return [grid_x, grid_y]
    

    def world_segment_to_grid_segment(self, segment:Segment) -> List[Tuple[int, int]]: 
        # Discretizes a line to the occupancy grid using Bresenham's Line algorithm. 
        # Source: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm 
        grid_segment_points = [] 
        start_segment = self._world_to_grid(segment.p1)
        end_segment = self._world_to_grid(segment.p2)
        

        dx, dy = abs(end_segment[0] - start_segment[0]), abs(end_segment[1] - start_segment[1])
        grid_x, grid_y = start_segment[0], start_segment[1] 
        sign_x = -1 if start_segment[0] > end_segment[0] else 1 
        sign_y = -1 if start_segment[1] > end_segment[1] else 1 
        error = dx - dy 

        while True: 
            grid_segment_points.append((grid_x, grid_y)) 
            if grid_x == end_segment[0] and grid_y == end_segment[1]: 
                break

            e2=2 * error 

            if e2 > -dy: 
                error -=dy 
                grid_x += sign_x 
                
            if e2 < dx: 
                error += dx 
                grid_y += sign_y 
        return grid_segment_points 


class CollisionChecker:
    """
    This class implements the collision check ability of a simple planner for a circular differential drive robot.

    Note that check_collision could be used to check collision between given GeoPrimitives
    check_collision function uses the functions that you implemented in CollisionPrimitives class.
    """

    def __init__(self):
        pass
    
    def path_collision_check(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """

        colliding_seg = []
        for i in range(len(t.waypoints)-1):
            collide = False
            x1,y1 = t.waypoints[i].x, t.waypoints[i].y
            x2,y2 = t.waypoints[i+1].x, t.waypoints[i+1].y

            seg_angle = np.arctan((y2-y1)/(x2-x1))
            shift_vec = r * np.array([-np.sin(seg_angle), np.cos(seg_angle)])

            # point_start = []
            # point_end = []
            for j in [-1,0,1]:
                point_start = Point(x1 + j*shift_vec[0], y1 + j*shift_vec[1])
                point_end = Point(x2 + j*shift_vec[0], y2 + j*shift_vec[1])

                seg = Segment(point_start, point_end)

                for obstacle in obstacles:
                    collide = check_collision(seg, obstacle)
                    if collide == True:
                        colliding_seg.append(i)
                        break

            if i not in colliding_seg:
                for obstacle in obstacles:
                    collide1 = check_collision(obstacle, Circle(Point(x1,y1),r))
                    collide2 = check_collision(obstacle, Circle(Point(x2,y2),r))
                    if collide1 == True or collide2 == True:
                            colliding_seg.append(i)
                            break
                    
        return sorted(set(colliding_seg))

    def path_collision_check_occupancy_grid(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will generate an occupancy grid of the given map.
        Then, occupancy grid will be used to check collisions.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """

        # create an occupanct grid with all obstacles
        # for each path segment transfer that to the grid and do collision check
        occ_grid_cells_x = 31
        occ_grid_cells_y = 31
        obstacles_shapely = []
        colliding_seg = []
        for obstacle in obstacles:
            obstacle_shapely = geo_primitive_to_shapely(obstacle)
            obstacles_shapely.append(obstacle_shapely)
            
        [min_x,min_y,max_x,max_y] = shapely.total_bounds(obstacles_shapely).tolist()

        #min_x, min_y is the origin
        grid_origin = Point(min_y, min_y)
        res_x = (max_x-min_x)/occ_grid_cells_x
        res_y = (max_y-min_y)/occ_grid_cells_y

        occ_grid = Grid(res_x, res_y, grid_origin, occ_grid_cells_x, occ_grid_cells_y, obstacles)

        for i in range(len(t.waypoints)-1):
            p1 = t.waypoints[i]
            p2 = t.waypoints[i+1]

            seg_angle = np.arctan((p2.y-p1.y)/(p2.x-p1.x))

            seg_angle = np.arctan2(p2.y-p1.y, p2.x-p1.x)
            shift_vec = r * np.array([-np.sin(seg_angle), np.cos(seg_angle)])
            shift_vec2 = -r * np.array([np.cos(seg_angle), np.sin(seg_angle)])

            point_start = Point(p1.x + shift_vec2[0], p1.y + shift_vec2[1])
            point_end = Point(p2.x - shift_vec2[0], p2.y - shift_vec2[1])
            grid_seg = occ_grid.world_segment_to_grid_segment(Segment(point_start, point_end))
            
            for j in [-1,1]:
                point_start = Point(p1.x + j*shift_vec[0], p1.y + j*shift_vec[1])
                point_end = Point(p2.x + j*shift_vec[0], p2.y + j*shift_vec[1])
                grid_seg = grid_seg + occ_grid.world_segment_to_grid_segment(Segment(point_start, point_end))

            for cell in grid_seg:
                if occ_grid.grid[cell] == 1:
                    colliding_seg.append(i)
                    break

        return sorted(set(colliding_seg))

    def path_collision_check_r_tree(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will build an R-Tree of the given obstacles.
        You are free to implement your own R-Tree or you could use STRTree of shapely module.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """
        obstacles_shapely = []
        for obstacle in obstacles:
            obstacles_shapely.append(geo_primitive_to_shapely(obstacle))


        r_tree = shapely.STRtree(obstacles_shapely)

        colliding_seg= []
        for i in range(len(t.waypoints)-1):
            collide = False
            x1,y1 = t.waypoints[i].x, t.waypoints[i].y
            x2,y2 = t.waypoints[i+1].x, t.waypoints[i+1].y

            seg_angle = np.arctan((y2-y1)/(x2-x1))
            shift_vec = r * np.array([-np.sin(seg_angle), np.cos(seg_angle)])


            p1 = Point(x1 + 1*shift_vec[0], y1 + 1*shift_vec[1])
            p2 = Point(x2 + 1*shift_vec[0], y2 + 1*shift_vec[1])
            p3 = Point(x2 + -1*shift_vec[0], y1 + -1*shift_vec[1])
            p4 = Point(x1 + -1*shift_vec[0], y1 + -1*shift_vec[1])

            seg_poly = Polygon([p1,p2,p3,p4])
            seg_poly_shapely = geo_primitive_to_shapely(seg_poly)

            aabb_obs_idx = r_tree.query(seg_poly_shapely)

            aabb_obs = [obstacles[i] for i in aabb_obs_idx.tolist()]
            path_seg = Path([t.waypoints[i], t.waypoints[i+1]])

            if len(self.path_collision_check(path_seg,r, aabb_obs)) >0:
                colliding_seg.append(i)
            # for obs in aabb_obs:
            #     if obs.intersects(seg_poly_shapely):
            #         colliding_seg.append(i)
            #         break

            if i not in colliding_seg:
                cir = Circle(t.waypoints[i], r)
                cir_shapely = geo_primitive_to_shapely(cir)
                aabb_obs_idx = r_tree.query(cir_shapely)
                for obs in [obstacles[i] for i in aabb_obs_idx.tolist()]:
                    if check_collision(obs,cir):
                        colliding_seg.append(i)
                        break
                # for obs in [obstacles_shapely[i] for i in aabb_obs_idx.tolist()]:
                #     if obs.intersects(cir_shapely):
                #         colliding_seg.append(i)
                #         break
            
            if i not in colliding_seg:
                cir = Circle(t.waypoints[i], r)
                cir_shapely = geo_primitive_to_shapely(cir)
                aabb_obs_idx = r_tree.query(cir_shapely)
                for obs in [obstacles[i] for i in aabb_obs_idx.tolist()]:
                    if check_collision(obs,cir):
                        colliding_seg.append(i)
                        break
                # for obs in [obstacles_shapely[i] for i in aabb_obs_idx.tolist()]:
                #     if obs.intersects(cir_shapely):
                #         colliding_seg.append(i)
                #         break

        return sorted(set(colliding_seg))

    def collision_check_robot_frame(
        self,
        r: float,
        current_pose: SE2Transform,
        next_pose: SE2Transform,
        observed_obstacles: List[GeoPrimitive],
    ) -> bool:
        """
        Returns there exists a collision or not during the movement of a circular differential drive robot until its next pose.

            Parameters:
                    r (float): Radius of circular differential drive robot
                    current_pose (SE2Transform): Current pose of the circular differential drive robot
                    next_pose (SE2Transform): Next pose of the circular differential drive robot
                    observed_obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives in robot frame
                    Please note that only Triangle, Circle and Polygon exist in this list
        """

        p1 = current_pose.p
        p2 = next_pose.p
        dist = np.linalg.norm(p1-p2)

        start = Point(0,0)
        end = Point(dist,0)

        path_seg = Path([start,end])

        if len(self.path_collision_check(path_seg, r, observed_obstacles))>0:
            return True
        return False

    def path_collision_check_safety_certificate(
        self, t: Path, r: float, obstacles: List[GeoPrimitive]
    ) -> List[int]:
        """
        Returns the indices of collided line segments.
        Note that index of first line segment is 0 and last line segment is len(t.waypoints)-1

        In this method, you will implement the safety certificates procedure for collision checking.
        You are free to use shapely to calculate distance between a point and a GoePrimitive.
        For more information, please check Algorithm 1 inside the following paper:
        https://journals.sagepub.com/doi/full/10.1177/0278364915625345.

            Parameters:
                    t (Path): Path of circular differential drive robot
                    r (float): Radius of circular differential drive robot
                    obstacles (List[GeoPrimitive]): List of obstacles as GeoPrimitives
                    Please note that only Triangle, Circle and Polygon exist in this list
        """

        obstacles_shapely = []
        for obstacle in obstacles:
            obstacles_shapely.append(geo_primitive_to_shapely(obstacle))

        # obstacles = obstacles_shapely
        safe_pts = {}
        colliding_seg = []
        for i in range(len(t.waypoints)-1):
            start_point = geo_primitive_to_shapely(t.waypoints[i])
            end_point = geo_primitive_to_shapely(t.waypoints[i+1])
            start = t.waypoints[i]
            end = t.waypoints[i+1]
            diff = [end.x-start.x, end.y-start.y]
            len_seg = np.linalg.norm(diff)

            d = -r
            nxt_pt = start_point
            nxt_pt_geo = start
            while(d<=len_seg):
                d_pt = np.inf
                for obstacle in obstacles_shapely:
                    if shapely.distance(obstacle, nxt_pt)< d_pt:
                        d_pt = shapely.distance(obstacle, nxt_pt)
                        
                d = d+d_pt
                if d_pt <= r:
                    colliding_seg.append(i)
                    break

                last_pt = nxt_pt_geo
                nxt_pt_geo = self._get_new_pt(last_pt, end, d_pt)
                nxt_pt = geo_primitive_to_shapely(nxt_pt_geo)
        return sorted(set(colliding_seg))

    def _get_new_pt(self,start: Point, end: Point, dist: float) -> Point:
        diff = [end.x-start.x, end.y-start.y]
        d = np.linalg.norm(diff)
        try:
            ratio = dist/d
        except:
            ratio = np.inf

        if ratio > 1:
            return end

        x = (1-ratio)*start.x + ratio*end.x
        y = (1-ratio)*start.y + ratio*end.y

        return Point(x,y)