# Libraries
import cv2
import numpy as np
import time
import threading
import math
import matplotlib.pyplot as plt

# Color constants (BGR)
RED = (0, 0, 255)
GREEN = (0, 200, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 255, 165)
PURPLE = (128, 0, 128)

# Global variables and their mutexes
DISPLAY_MAP = np.array((1, 1)).astype(np.float32)
DISPLAY_MAP_2 = np.array((1, 1)).astype(np.float32)
EXIT_SIGNAL = False
worker_mutex = threading.Lock()
worker_mutex_2 = threading.Lock()
exit_signal_mutex = threading.Lock()

class Node:

    """
    Class that represents each node of the grid
    """

    def __init__(self, position, parent=None):

        # Position of the node
        self.position = position
        # Parent of the node -> previous in the path
        self.parent = parent

        # variables of the A* algorithm
        # f = g + h
        self.g = 0
        self.h = 0
        self.f = 0


    def __eq__(self, other_node):
        return self.position == other_node.position

# Dict that contains all the posible movements
movements_dict = {
    "up": (0, -1),
    "down": (0, 1),
    "right": (1, 0),
    "left": (-1, 0),
    "up_right": (1, -1),
    "up_left": (-1, -1),
    "down_right": (1, 1),
    "down_left": (-1, 1)
}

class Robot:

    """
    Class that represents the robot
    """

    def __init__(self, start_pos, target_pos, map, delay=0.005, radius=5, heuristic="manhattam"):

        """
        Constructor: Initializes all the elements
        """

        self.start_pos = start_pos
        self.current_pos = start_pos
        self.target_pos = target_pos
        self.radius = radius
        self.map = map

        # Copies of the map that are going to be painted
        self.painted_map = self.map.copy()
        self.position_map = self.map.copy()

        # Check map is a copy of the map where whether a cell contains
        # an obstacle or not is checked. When a robot is bigger than one
        # cell, obstacles are dilated so that the robot can be treated
        # as a single point
        if self.radius > 1:
            self.check_map = cv2.erode(self.map.copy(), np.ones((self.radius*2+1,
                                                          self.radius*2+1), np.uint8))
        else:
            self.check_map = self.map.copy()

        self.delay = delay

        # Heuristic used in the A*
        self.heuristic = heuristic

        self.path = []
        self.draw_target_pos(self.position_map)
        self.draw_start_pos(self.position_map)

    def draw_target_pos(self, map, draw=True):

        """
        Method that draws the target position of the robot
        """

        global DISPLAY_MAP, worker_mutex

        if self.radius > 1:
            cv2.circle(map, self.target_pos, self.radius, RED, -1)
        else:
            map[self.target_pos[1], self.target_pos[0], :] = RED


        if draw:
            worker_mutex.acquire()
            DISPLAY_MAP = map
            worker_mutex.release()

    def draw_start_pos(self, map, draw=True):

        """
        Method that draws the start position of the robot
        """

        global DISPLAY_MAP, worker_mutex

        if self.radius > 1:
            cv2.circle(map, self.start_pos, self.radius, PURPLE, -1)
        else:
            map[self.start_pos[1], self.start_pos[0], :] = PURPLE

        if draw:
            worker_mutex.acquire()
            DISPLAY_MAP = map
            worker_mutex.release()

    def draw_position(self):

        """
        Method that draws the current position of the robot
        """

        global DISPLAY_MAP, worker_mutex

        self.position_map[self.current_pos[1], self.current_pos[0], :] = BLUE
        map = self.position_map.copy()

        if self.radius > 1:
            cv2.circle(map, self.current_pos, self.radius, RED, -1)
        else:
            map[self.current_pos[1], self.current_pos[0], :] = RED


        worker_mutex.acquire()
        DISPLAY_MAP = map
        worker_mutex.release()

    def set_closed(self, node):

        """
        Method that draws a node as member of the closed set
        in its corresponding map
        """

        global DISPLAY_MAP_2, worker_mutex_2

        map = self.painted_map
        map[node.position[1], node.position[0], :] = BLUE

        self.draw_target_pos(map, draw=False)
        self.draw_start_pos(map, draw=False)

        worker_mutex_2.acquire()
        DISPLAY_MAP_2 = map
        worker_mutex_2.release()

    def set_opened(self, node):

        """
        Method that draws a node as member of the open set
        in it
        """

        global DISPLAY_MAP_2, worker_mutex_2

        map = self.painted_map
        map[node.position[1], node.position[0], :] = GREEN

        self.draw_target_pos(map, draw=False)
        self.draw_start_pos(map, draw=False)

        worker_mutex_2.acquire()
        DISPLAY_MAP_2 = map
        worker_mutex_2.release()
    

    def plan_trajectory(self):

        """
        Method that plans the optimal trajectory between start and target
        positions using the A* algorithm
        """

        # Creates the sets
        open_set = []
        closed_set = []

        # Creates both start and target node
        start_node = Node(position=self.start_pos)
        target_node = Node(position=self.target_pos)

        # Appends the current node to the open set 
        open_set.append(start_node)
        self.set_opened(start_node)

        print("[I] Planning optimal trajectory")

        # While the open set contains nodes
        while len(open_set) > 0:

            # Selects the first one (they are ordered) according to f)
            # appends it to the closed set and analises it
            current_node = open_set.pop(0)
            closed_set.append(current_node)
            self.set_closed(current_node)

            # If current node is target node the path has been found
            if current_node.position == target_node.position:

                print("[I] Optimal path found!")
                print(f"[I] Number of analized points: {len(closed_set)}")
                self.path = []
                current = current_node
                while current is not None:
                    self.path.append(current)
                    current = current.parent

                # Reverses the path because it has been built backwards
                self.path.reverse()
                return

            # Loop that analises the neightborhood (children) of the current node
            children = []
            for key in movements_dict:

                children_pos = (current_node.position[0] + movements_dict[key][0],
                                current_node.position[1] + movements_dict[key][1])


                # It the childs position is not legal go to the next one
                if children_pos[0] < 0 or children_pos[0] > self.map.shape[1] or\
                   children_pos[1] < 0 or children_pos[1] > self.map.shape[0]:
                    continue

                # If the child node contains an obstacle go to the next one
                if np.all(self.check_map[children_pos[1], children_pos[0]] == 0):
                    continue

                # Instantiate the childs node
                new_node = Node(position=children_pos, parent=current_node)

                # If the movemente has been diagonal (they have "_" in movements_dict) increase
                # g by 1.4, else by 1
                if "_" in key:
                    new_node.g = current_node.g + 1.4142
                else:
                    new_node.g = current_node.g + 1

                # Computes its h and f
                new_node.h = heuristic(new_node, target_node, heuristic=self.heuristic) 
                new_node.f = new_node.g + new_node.h

                # Consider it as a candidate to the open set
                children.append(new_node)

            # For each child candidate to the open set
            for child in children:

                # If it already is on the closed set do nothing
                if len([closed_child for closed_child in closed_set if closed_child.position == child.position]) > 0:
                    continue

                # If it already is in the open set and also has a better value of g do nothing
                if len([open_node for open_node in open_set if child.position == open_node.position and child.g >= open_node.g]) > 0:
                    continue

                # Append it to the open set
                open_set.append(child)
                self.set_opened(child)

            # Sort the open set by f
            open_set = sorted(open_set, key=lambda Node: Node.f)

            # Just for visualization
            time.sleep(self.delay)

            # Checks the exit signal (ESC key)
            exit_signal_mutex.acquire()
            if EXIT_SIGNAL:
                break
            exit_signal_mutex.release()


        print("[I] Cannot find optimal path")

    def navigate_planned_path(self):

        """
        Method that represents the navigation of the 
        robot graphically
        """

        if len(self.path) > 1:
            last_node = self.path[0]
            # For each node in path
            for p in self.path:
                # Checks the exit signal (ESC key)
                exit_signal_mutex.acquire()
                if EXIT_SIGNAL:
                    break
                exit_signal_mutex.release()

                # Draws its position
                self.current_pos = p.position
                self.draw_position()

                # Backups the last node so as to keep its the lenght of the path -> final_node.g
                last_node = p
                time.sleep(self.delay*25)

            print(f"[I] Length of the found path: {last_node.g:.2f} cells")


def heuristic(node, target_node, heuristic="euclidean"):

    """
    Function that computes the heuristic given current and target nodes
    """

    if heuristic == "euclidean":
        h = math.sqrt((np.float32(node.position[0]) -
                    np.float32(target_node.position[0]))**2 +
                    (np.float32(node.position[1]) -
                    np.float32(target_node.position[1]))**2)
    elif heuristic == "manhattam":
        h = np.abs(node.position[0] -
                   target_node.position[0]) +\
            np.abs(node.position[1] -
                   target_node.position[1])

    return h


def main():

    """
    Main function
    """

    # Reads the map file
    map = read_map()

    # Creates the robot object
    my_robot = Robot((8, 70), (80, 17), map, radius=1, heuristic="euclidean") #manhattam, euclidean

    # Plans the trajectory
    my_robot.plan_trajectory()
    # Draw when finished
    my_robot.navigate_planned_path()

    exit_signal_mutex.acquire()
    if not EXIT_SIGNAL:
        save_imgs()
    exit_signal_mutex.release()


def read_map(map_file="office.tif"):

    """
    Reads the map saved as a tif file
    """

    return cv2.imread(f"../images/{map_file}")


def display_worker():

    """
    Function called on other thread. It manages graphical representations
    """

    global DISPLAY_MAP, worker_mutex
    global DISPLAY_MAP_2, worker_mutex_2
    global EXIT_SIGNAL, exit_signal_mutex

    cv2.namedWindow("Map navigation", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    while not EXIT_SIGNAL:
        
        worker_mutex.acquire()
        worker_mutex_2.acquire()
        navigation_map = np.vstack([DISPLAY_MAP.copy(), DISPLAY_MAP_2.copy()])
        worker_mutex_2.release()
        worker_mutex.release()

        cv2.imshow("Map navigation", navigation_map)

        if cv2.waitKey(33) == 27:
            exit_signal_mutex.acquire()
            EXIT_SIGNAL = True
            exit_signal_mutex.release()
            break

    cv2.destroyAllWindows()

def save_imgs():

    """
    Function that saves images to disk
    """

    global DISPLAY_MAP, worker_mutex
    global DISPLAY_MAP_2, worker_mutex_2

    worker_mutex.acquire()
    worker_mutex_2.acquire() 

    cv2.imwrite("path.png", DISPLAY_MAP)
    cv2.imwrite("cells.png", DISPLAY_MAP_2)

    worker_mutex_2.release()
    worker_mutex.release()


if __name__ == '__main__':
    try:
        display_thread = threading.Thread(target=display_worker)
        display_thread.start()
        main()
        
    except KeyboardInterrupt:
        exit()
