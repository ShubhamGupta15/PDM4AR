from typing import Sequence

from dg_commons import SE2Transform

from math import *

from pdm4ar.exercises.ex05.structures import *
from pdm4ar.exercises_def.ex05.utils import extract_path_points


class PathPlanner(ABC):
    @abstractmethod
    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        pass


class Dubins(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> List[SE2Transform]:
        """ Generates an optimal Dubins path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow
        """
        path = calculate_dubins_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


class ReedsShepp(PathPlanner):
    def __init__(self, params: DubinsParam):
        self.params = params

    def compute_path(self, start: SE2Transform, end: SE2Transform) -> Sequence[SE2Transform]:
        """ Generates a Reeds-Shepp *inspired* optimal path between start and end configuration

        :param start: the start configuration of the car (x,y,theta)
        :param end: the end configuration of the car (x,y,theta)

        :return: a List[SE2Transform] of configurations in the optimal path the car needs to follow 
        """
        path = calculate_reeds_shepp_path(start_config=start, end_config=end, radius=self.params.min_radius)
        se2_list = extract_path_points(path)
        return se2_list


def calculate_car_turning_radius(wheel_base: float, max_steering_angle: float) -> DubinsParam:
    turn_radius = wheel_base/tan(max_steering_angle)
    return DubinsParam(min_radius=turn_radius)


def calculate_turning_circles(current_config: SE2Transform, radius: float) -> TurningCircle:
    theta = current_config.theta
    pos_vec = current_config.p
    rad_vec = np.array((sin(theta), -cos(theta)))
    #left circle
    center = pos_vec - rad_vec*radius
    left_circle = Curve.create_circle(center=SE2Transform(p = center, theta = 0), config_on_circle=current_config,
                                       radius=radius, curve_type=DubinsSegmentType.LEFT)  
    
    #right circle
    center = pos_vec + rad_vec*radius
    right_circle = Curve.create_circle(center=SE2Transform(p = center, theta = 0), config_on_circle=current_config,
                                       radius=radius, curve_type=DubinsSegmentType.RIGHT)  
    

    return TurningCircle(left=left_circle, right=right_circle)


def calculate_tangent_btw_circles(circle_start: Curve, circle_end: Curve) -> List[Line]:
    tangents = []

    center_dist = np.linalg.norm(circle_end.center.p-circle_start.center.p, ord=2)
    
    if center_dist == 0:
        # overlapping circles ====> no tangents
        tangents = [] 


    elif (circle_end.radius+circle_start.radius) > center_dist:
        # intersecting circles ====> two possible tangets
        if circle_end.type != circle_start.type:
            tangents = []
        else:
            tangents = [parallel_tangent(circle_start, circle_end)]


    elif (circle_end.radius+circle_start.radius) == center_dist:
        if circle_end.type != circle_start.type:
            tangents = [cross_tangent(circle_start, circle_end)]
        else:
            tangents = [parallel_tangent(circle_start, circle_end)]


    elif (circle_end.radius+circle_start.radius) < center_dist:
        if circle_end.type != circle_start.type:
            tangents = [cross_tangent(circle_start, circle_end)]
        else:
            tangents = [parallel_tangent(circle_start, circle_end)]
    return tangents  


def calculate_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:

    turning_circles_start = calculate_turning_circles(start_config, radius)
    turning_circles_end = calculate_turning_circles(end_config, radius)

    best_path = []
    best_len = np.inf
    for i in ['L','R']:
        for j in ['L', 'R']:
            segment = []
            if i == 'L':
                circle_start = turning_circles_start.left
            else:
                circle_start = turning_circles_start.right
            if j == 'L':
                circle_end = turning_circles_end.left
            else:
                circle_end = turning_circles_end.right
    
            tangent = calculate_tangent_btw_circles(circle_start, circle_end)
            if len(tangent) == 0:
                # no possible path for this config
                continue
            
            angle_start = calculate_arc_length(start_config, tangent[0].start_config, circle_start.type)
            angle_end = calculate_arc_length(tangent[0].end_config, end_config, circle_end.type)

            segment.append(Curve (start_config=start_config, end_config=tangent[0].start_config, center= circle_start.center, 
                                   radius= circle_start.radius, curve_type=circle_start.type,arc_angle= angle_start ))
            segment.append(tangent[0])
            segment.append(Curve (start_config=tangent[0].end_config, end_config=end_config, center= circle_end.center, 
                                   radius= circle_end.radius, curve_type=circle_end.type,arc_angle= angle_end ))
            path_len =segment[0].arc_angle*segment[0].radius + segment[1].length +segment[2].arc_angle*segment[2].radius

            if path_len < best_len:
                best_len = path_len
                best_path = segment

    if np.linalg.norm(start_config.p - end_config.p) < 6*radius:
        ##LRL
        LRL_path = calculate_LRL_path(start_config, end_config, radius)
        if len(LRL_path) !=0:
            path_len =LRL_path[0].arc_angle*LRL_path[0].radius + LRL_path[1].arc_angle*LRL_path[1].radius +LRL_path[2].arc_angle*LRL_path[2].radius 
            if path_len<best_len:
                best_len =  path_len 
                best_path = LRL_path

        RLR_path = calculate_RLR_path(start_config, end_config, radius)
        if len(RLR_path) != 0:
            path_len =RLR_path[0].arc_angle*RLR_path[0].radius + RLR_path[1].arc_angle*RLR_path[1].radius +RLR_path[2].arc_angle*RLR_path[2].radius

            if path_len<best_len:
                best_len = path_len
                best_path = RLR_path
        
    return best_path


def calculate_reeds_shepp_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:
    
    path_dubins = calculate_dubins_path(start_config, end_config, radius)
    path_dubins_len = path_dubins[0].arc_angle*path_dubins[0].radius + path_dubins[1].length +path_dubins[2].arc_angle*path_dubins[2].radius

    # reverse_start_config = SE2Transform(start_config.p, start_config.theta + np.pi)
    # reverse_end_config = SE2Transform(end_config.p, end_config.theta + np.pi)

    path_reverse = calculate_reverse_dubins_path(rc(start_config), rc(end_config), radius)

    path_rev_len = path_reverse[0].arc_angle*path_reverse[0].radius + path_reverse[1].length +path_reverse[2].arc_angle*path_reverse[2].radius
    # path_rev_len = np.inf
    if path_dubins_len < path_rev_len :
        return path_dubins
    else:
        return path_reverse


# utils
def calculate_arc_length(start_config, end_config, curve_type):
    theta = end_config.theta - start_config.theta
    if theta<0 and curve_type == DubinsSegmentType.LEFT:
        theta = theta + 2*np.pi
    elif theta>0 and curve_type == DubinsSegmentType.RIGHT:
        theta = theta - 2*np.pi
    
    return abs(theta)

def parallel_tangent(circle_start: Curve, circle_end: Curve):
    start_end_vec = circle_end.center.p - circle_start.center.p
    start_end_theta = np.arctan2(start_end_vec[1], start_end_vec[0])

    rad_vec = circle_start.radius * np.array((sin(start_end_theta), -cos(start_end_theta)))

    if circle_start.type == DubinsSegmentType.LEFT:
        point_start = circle_start.center.p + rad_vec
        point_end = circle_end.center.p + rad_vec
    else:
        point_start = circle_start.center.p - rad_vec
        point_end = circle_end.center.p - rad_vec

    return Line(start_config=SE2Transform(point_start, start_end_theta), end_config=SE2Transform(point_end, start_end_theta))

def cross_tangent(circle_start: Curve, circle_end: Curve):
    start_end_vec = circle_end.center.p - circle_start.center.p
    start_end_len = np.linalg.norm(start_end_vec)

    start_end_theta = np.arctan2(start_end_vec[1], start_end_vec[0])
    rot_theta = np.arccos((2*circle_start.radius)/start_end_len)  ################## faliure point

    if circle_start.type == DubinsSegmentType.LEFT:
        tot_theta = start_end_theta - rot_theta
    else:
        tot_theta = start_end_theta + rot_theta

    rad_vec = circle_start.radius * np.array((cos(tot_theta), sin(tot_theta)))

    point_start = circle_start.center.p + rad_vec
    point_end = circle_end.center.p - rad_vec

    line_vec = point_end-point_start
    line_theta = np.arctan2(line_vec[1],line_vec[0])

    return Line(start_config=SE2Transform(point_start, line_theta), end_config=SE2Transform(point_end, line_theta))



def calculate_reverse_dubins_path(start_config: SE2Transform, end_config: SE2Transform, radius: float) -> Path:

    turning_circles_start = calculate_turning_circles(start_config, radius)
    turning_circles_end = calculate_turning_circles(end_config, radius)

    best_path = []
    best_len = np.inf
    for i in ['L','R']:
        for j in ['L', 'R']:
            segment = []
            if i == 'L':
                circle_start = turning_circles_start.left
            else:
                circle_start = turning_circles_start.right
            if j == 'L':
                circle_end = turning_circles_end.left
            else:
                circle_end = turning_circles_end.right
    
            tangent = calculate_tangent_btw_circles(circle_start, circle_end)
            if len(tangent) == 0:
                # no possible path for this config
                continue
            
            tangent = tangent[0]
            angle_start = calculate_arc_length(start_config, tangent.start_config, circle_start.type)
            angle_end = calculate_arc_length(tangent.end_config, end_config, circle_end.type)

            segment.append(Curve (start_config=rc(start_config), end_config=rc(tangent.start_config), center= circle_start.center, 
                                   radius= circle_start.radius, curve_type=circle_start.type,arc_angle= angle_start, gear=Gear.REVERSE ))
            segment.append(Line(start_config=rc(tangent.start_config), end_config= rc(tangent.end_config), gear= Gear.REVERSE))
            segment.append(Curve (start_config=rc(tangent.end_config), end_config=rc(end_config), center= circle_end.center, 
                                   radius= circle_end.radius, curve_type=circle_end.type,arc_angle= angle_end, gear=Gear.REVERSE ))
            
            path_len =segment[0].arc_angle*segment[0].radius + segment[1].length +segment[2].arc_angle*segment[2].radius

            if path_len < best_len:
                best_len = path_len
                best_path = segment

    return best_path

def rc(config:SE2Transform) -> SE2Transform:
    #reverse configuration
    return SE2Transform(config.p, config.theta + np.pi)


def calculate_LRL_path(start_config, end_config, radius):
    LRL_path = []
    turning_circles_start = calculate_turning_circles(start_config, radius)
    turning_circles_end = calculate_turning_circles(end_config, radius)

    circle_start = turning_circles_start.left
    circle_end = turning_circles_end.left

    start_end_vec = circle_end.center.p - circle_start.center.p
    start_end_vec_len = np.linalg.norm(start_end_vec)
    start_end_theta = np.arctan2(start_end_vec[1], start_end_vec[0])

    try:
        rot_theta = np.arccos(start_end_vec_len/(4*radius))
    except Exception:
        return []


    circle_tangent_center = circle_start.center.p + 2*radius*np.array([cos(start_end_theta+rot_theta), sin(start_end_theta+rot_theta)])

    start_circle_tangent_pt = circle_start.center.p + radius*np.array([cos(start_end_theta+rot_theta), sin(start_end_theta+rot_theta)])

    end_circle_tanget_pt = circle_tangent_center + radius*(circle_end.center.p - circle_tangent_center)/np.linalg.norm(circle_end.center.p - circle_tangent_center)

    p1_angle = np.pi/2 + start_end_theta + rot_theta
    vec = circle_end.center.p - circle_tangent_center
    p2_angle = -np.pi/2+ np.arctan2(vec[1],vec[0])

    p1_config = SE2Transform(start_circle_tangent_pt, p1_angle)
    p2_config = SE2Transform(end_circle_tanget_pt, p2_angle)

    angle_start = calculate_arc_length(start_config, p1_config,DubinsSegmentType.LEFT )
    angle_mid = calculate_arc_length(p1_config,p2_config, DubinsSegmentType.RIGHT)
    angle_end = calculate_arc_length(p2_config, end_config, DubinsSegmentType.LEFT)

    LRL_path.append(Curve(start_config=start_config, end_config=p1_config,center=circle_start.center, radius = radius,
                            curve_type= DubinsSegmentType.LEFT, arc_angle=angle_start))
    LRL_path.append(Curve(start_config=p1_config, end_config=p2_config, center=SE2Transform(circle_tangent_center,0), radius=radius,
                            curve_type=DubinsSegmentType.RIGHT, arc_angle=angle_mid))
    LRL_path.append(Curve(start_config=p2_config, end_config=end_config, center=circle_end.center, radius = radius,
                            curve_type=DubinsSegmentType.LEFT, arc_angle=angle_end))
    
    
    return LRL_path



def calculate_RLR_path(start_config, end_config, radius):
    RLR_path = []
    turning_circles_start = calculate_turning_circles(start_config, radius)
    turning_circles_end = calculate_turning_circles(end_config, radius)

    circle_start = turning_circles_start.right
    circle_end = turning_circles_end.right

    start_end_vec = circle_end.center.p - circle_start.center.p
    start_end_vec_len = np.linalg.norm(start_end_vec)
    start_end_theta = np.arctan2(start_end_vec[1], start_end_vec[0])

    try:
        rot_theta = np.arccos(start_end_vec_len/(4*radius))
    except:
        return []


    circle_tangent_center = circle_start.center.p + 2*radius*np.array([cos(start_end_theta-rot_theta), sin(start_end_theta-rot_theta)])

    start_circle_tangent_pt = circle_start.center.p + radius*np.array([cos(start_end_theta+rot_theta), sin(start_end_theta+rot_theta)])

    end_circle_tanget_pt = circle_tangent_center + radius*(circle_end.center.p - circle_tangent_center)/np.linalg.norm(circle_end.center.p - circle_tangent_center)

    p1_angle = -np.pi/2 + start_end_theta + rot_theta
    vec = circle_end.center.p - circle_tangent_center
    p2_angle = np.pi/2+ np.arctan2(vec[1],vec[0])

    p1_config = SE2Transform(start_circle_tangent_pt, p1_angle)
    p2_config = SE2Transform(end_circle_tanget_pt, p2_angle)

    angle_start = calculate_arc_length(start_config, p1_config,DubinsSegmentType.RIGHT )
    angle_mid = calculate_arc_length(p1_config,p2_config, DubinsSegmentType.LEFT)
    angle_end = calculate_arc_length(p2_config, end_config, DubinsSegmentType.RIGHT)

    RLR_path.append(Curve(start_config=start_config, end_config=p1_config,center=circle_start.center, radius = radius,
                            curve_type= DubinsSegmentType.RIGHT, arc_angle=angle_start))
    RLR_path.append(Curve(start_config=p1_config, end_config=p2_config, center=SE2Transform(circle_tangent_center,0), radius=radius,
                            curve_type=DubinsSegmentType.LEFT, arc_angle=angle_mid))
    RLR_path.append(Curve(start_config=p2_config, end_config=end_config, center=circle_end.center, radius = radius,
                            curve_type=DubinsSegmentType.RIGHT, arc_angle=angle_end))
    
    return RLR_path



