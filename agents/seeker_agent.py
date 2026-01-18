from agent_base import Agent
from world_state import WorldState
import math
import heapq
import random
from Mesh.nav_mesh import NavMesh, NavMeshCell
import shapely
import numpy as np




class DumbSeeker(Agent):


   def __init__(self, world_map: NavMesh, max_speed: float):
       Agent.__init__(self, world_map, max_speed)
       self.name = "Dumb Seeker"
       self._cell_path = None
       self._waypoints = None
       self._last_target = None
       self._roam_target = None
       self.stuck = False
       self.last_move = None
       self.seeker_last_pos = None
      
     
  
   def astar(self, start_cell: "NavMeshCell", end_cell: "NavMeshCell") -> list["NavMeshCell"] | None:
       """
   Find the shortest path between two cells using the A* pathfinding algorithm
   Do not access any variable that starts with an underscore, use the properties instead.
   You can access a cell's coordinates with cell.coord and its neighbors with cell.neighbors.
   If you need to get the distance between two cells, use cell.distance(other_cell)


   Args:
       start_cell (NavMeshCell): The starting location
       end_cell (NavMeshCell): The ending location


   Returns:
       list[NavMeshCell] | None: A list of NavMeshCells (representing the shortest path) or None
           if there is no valid path between the two locations.
   """
       def heurstic(currCell: NavMeshCell) -> float:
           return currCell.distance(end_cell)
       open = []
       came_from = {}
       g = {start_cell: 0}
       closed = set()
       heapq.heappush(open, (heurstic(start_cell),0,start_cell))
      
       while open:
           _, _, curr = heapq.heappop(open)


           if curr in closed:
               continue
           closed.add(curr)
          
           if curr == end_cell:
               path = [curr]
               while curr in came_from:
                   curr = came_from[curr]
                   path.append(curr)
               path.reverse()
               return path
          
           for neighbor in curr.neighbors.keys():
               if neighbor in closed:
                   continue
               possible_g = g[curr] + curr.distance(neighbor)
               if possible_g < g.get(neighbor, float("inf")):
                   came_from[neighbor] = curr
                   g[neighbor] = possible_g
                   f = possible_g + heurstic(neighbor)
                   count = random.randint(1, 1000)
                   heapq.heappush(open, (f, count, neighbor))


       return None


   def can_move(self, pos, dx, dy, margin=3.0):
       collider = self.map.polygon.buffer(margin)
       next_pos = shapely.affinity.translate(pos, dx, dy)
       line = shapely.LineString([pos, next_pos])
       return collider.contains(line)
  
   def try_turn(self, pos, dx, dy, step, margin=3.0):
       for deg in (15, -15, 30 , -30, 45, -45, 60, -60, 90, - 90, 120, -120, 150 , -150, 180):
           ang = math.radians(deg)
           ndx = dx*math.cos(ang) - dy*math.sin(ang)
           ndy = dy*math.sin(ang) + dy*math.cos(ang)


           norm = (ndx*ndx + ndy*ndy) ** 0.5
           if norm < 1e-6:
               continue
           ndx, ndy = ndx/norm * step, ndy/norm * step
           if self.can_move(pos, ndx, ndy, margin):
               return ndx, ndy
       return None
  
   def act(self, state: WorldState) -> tuple[float, float] | None:
       print("DumbSeeker act tick")
       seeker_pos = state.seeker_position
       goal_pos = state.hider_position


       if state is None or seeker_pos is None:
           return None


       if goal_pos is None or seeker_pos.equals(goal_pos):
           if self._roam_target is None or seeker_pos.distance(self._roam_target) < 6.0:
               self._roam_target = self.map.random_position()
               self._cell_path = None
               self._waypoints = None
               self._last_target = None
           goal_pos = self._roam_target
      
      
       if not self.map.in_bounds(goal_pos):
           self._roam_target = None
           return None


       start_cell = self.map.find_cell(seeker_pos)
       end_cell = self.map.find_cell(goal_pos)
       if not start_cell or not end_cell:
           return random.randint(-1,1), random.randint(-1,1)


       need_new_path = (
           self._cell_path is None
           or self._last_target is None
           or not self._last_target.equals(goal_pos)
           or not self._cell_path
           or self._cell_path[-1] != end_cell
           or start_cell not in self._cell_path
           or self.stuck == True
       )
       self.stuck = False
       if need_new_path:
           path = self.astar(start_cell, end_cell)
           if not path:
               self._cell_path = None
               self._waypoints = None
               return None
           self._cell_path = path
           self._last_target = goal_pos
       else:
           while (
               self._cell_path
               and len(self._cell_path) > 1
               and self._cell_path[1].polygon.covers(seeker_pos)
           ):
               self._cell_path.pop(0)


           if self._cell_path and self._cell_path[0] != start_cell:
               try:
                   idx = self._cell_path.index(start_cell)
                   self._cell_path = self._cell_path[idx:]
               except ValueError:
                   self._cell_path = None
                   self._waypoints = None
                   return None


       if not self._cell_path:
           self._waypoints = None
           return None


       while (
           self._cell_path
           and len(self._cell_path) > 1
           and self._cell_path[1].polygon.covers(seeker_pos)
       ):
           self._cell_path.pop(0)


       if not self._cell_path:
           self._waypoints = None
           return None


       portal_points: list[shapely.Point] = []
       for idx in range(len(self._cell_path) - 1):
           next_cell = self._cell_path[idx + 1]
           portal_points.append(next_cell.polygon.centroid)
       portal_points.append(goal_pos)


       self._waypoints = []
       anchor = seeker_pos
       index = 0
       while index < len(portal_points):
           far = index
           while far + 1 < len(portal_points) and self.map.has_line_of_sight(
               anchor, portal_points[far + 1]
           ):
               far += 1
           self._waypoints.append(portal_points[far])
           anchor = portal_points[far]
           index = far + 1


       if not self._waypoints:
           return None


       clear_distance = max(1.0, self.max_speed * 0.5)
       while (
           len(self._waypoints) > 1
           and seeker_pos.distance(self._waypoints[0]) <= clear_distance
       ):
           self._waypoints.pop(0)


       if not self._waypoints:
           self._cell_path = None
           return None


       waypoint = self._waypoints[0]
       if (
           len(self._waypoints) > 1
           and seeker_pos.distance(waypoint) <= clear_distance
           and self.map.has_line_of_sight(seeker_pos, self._waypoints[1])
       ):
           waypoint = self._waypoints[1]
       dx = waypoint.x - seeker_pos.x
       dy = waypoint.y - seeker_pos.y
       distance = math.hypot(dx, dy)
       if distance == 0:
           return None
       speed = min(distance, self.max_speed) -.01
       scale = speed / distance
       if not self.can_move( seeker_pos, dx* scale, dy*scale):
           print("stuck")
           ndx,ndy = self.try_turn(seeker_pos,dx*scale,dy*scale,9.9)
           if  ndx != None:
               return ndx, ndy
           self.stuck = True
           return (dx * scale, dy * scale)
       print(dx*scale,dy*scale)
       if seeker_pos == self.seeker_last_pos:
           self.stuck = True
       self.seeker_last_pos = seeker_pos
       print(dx * scale, dy * scale)
       return (dx * scale, dy * scale)


   @property
   def is_seeker(self) -> bool:
       return True
