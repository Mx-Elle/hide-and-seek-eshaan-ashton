#Eshaan

import math
from heapq import heappop, heappush
from typing import Optional
import shapely
from agent_base import Agent
from world_state import WorldState
from Mesh.nav_mesh import NavMesh, NavMeshCell

# A*
def astar(start_cell: NavMeshCell, end_cell: NavMeshCell) -> list[NavMeshCell] | None:
    #set up frontier as a list of tuples that include the priority value, a counter for tie-breaking, and the NavMeshCell
    frontier: list[tuple[float, int, NavMeshCell]] = []

    #essentially, the counter is just what is going to be compared when priorities are equal
    counter = 1
    heappush(frontier, (0.0, counter, start_cell))
    came_from = dict()
    cost_so_far = dict()
    came_from[start_cell] = None
    cost_so_far[start_cell] = 0

    #loop through frontier as long as it isn't empty
    while frontier:
        #discard priority and count values of this cell in the frontier because we only need the cell itself
        _priority, _count, current = heappop(frontier)

        #if we've reached the end_cell, we know that this is the shortest path because we found it first
        #go through came_from and add the path of cells to the list, eventually reversing it because we start from end_cell
        if current == end_cell:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbor in current.neighbors:

            #find the cost to get from start to the neighbor
            new_cost = cost_so_far[current] + current.distance(neighbor)

            #check if the neighbor either hasn't been visited or if it is cheaper to go there
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]: 
                cost_so_far[neighbor] = new_cost

                #priority is the new cost added to the heuristic distance
                priority = new_cost + neighbor.distance(end_cell) # f = g + h
                counter+=1

                #add this neighbor to the frontier heapk
                heappush(frontier,(priority, counter, neighbor))

                #add the current cell to the came_from dict 
                came_from[neighbor] = current
    return None


class DumbHider(Agent):
    """
    Logic depends on cached paths, portal waypoints, LOS shortcutting.
    target decision-making depends on distance, branchiness (how many paths to escape exist), out of line of sight cells.
    """

    def __init__(self, world_map: NavMesh, max_speed: float):
        Agent.__init__(self, world_map, max_speed)
        self.name = "Dumb Hider"
        self.map = world_map
        # Cache navmesh geometry to reduce the amount of work involving shapely per frame
        self._cells = list(self.map.cells)
        self._cell_centroid: dict[NavMeshCell, shapely.Point] = {
            cell: cell.polygon.centroid for cell in self._cells
        }
        self._cell_branch: dict[NavMeshCell, int] = {cell: len(cell.neighbors) for cell in self._cells}
        boundary = self.map.polygon.boundary
        self._cell_boundary_dist: dict[NavMeshCell, float] = {
            cell: boundary.distance(self._cell_centroid[cell]) for cell in self._cells
        }
        self._safe_polys: dict[float, shapely.Polygon] = {}
        self._map_centroid = self.map.polygon.centroid
        # A* cells and waypoints
        self._cell_path: list[NavMeshCell] | None = None
        self._path_target: NavMeshCell | None = None
        self._waypoints: list[shapely.Point] | None = None
        self._target_cell: NavMeshCell | None = None
        self._last_target_tick = 0
        self._target_recalc_interval = 8
        self._last_seeker_cell: NavMeshCell | None = None
        # Movement continuity and stuck recovery
        self._last_move: Optional[tuple[float, float]] = None
        self._last_state_sig: tuple[float, float, float, float] | None = None
        self._prev_hider_pos: shapely.Point | None = None
        self._stuck_counter = 0
        self._stuck_reset_cooldown = 0
        self._tick = 0
        # Head start detection
        self._initial_seeker_pos: shapely.Point | None = None
        self._seeker_started = False

    # Main decision loop
    def act(self, state: WorldState) -> tuple[float, float] | None:
        if not state or not state.hider_position:
            return None

        # Track state changes to reuse the last move when nothing changed.
        hider_pos = state.hider_position
        seeker_pos = state.seeker_position

        sig = (hider_pos.x, hider_pos.y, seeker_pos.x, seeker_pos.y)
        if sig == self._last_state_sig and self._last_move is not None:
            return self._last_move
        self._last_state_sig = sig
        self._tick += 1

        # Detect when stuck
        if self._prev_hider_pos is not None:
            if hider_pos.distance(self._prev_hider_pos) < 0.2:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
        self._prev_hider_pos = hider_pos

        # Detect when the seeker starts moving
        if self._initial_seeker_pos is None:
            self._initial_seeker_pos = seeker_pos
        elif not self._seeker_started and seeker_pos.distance(self._initial_seeker_pos) > 1e-3:
            self._seeker_started = True

        # Get hider and seeker cells
        start_cell = self.map.find_cell(hider_pos)
        seeker_cell = self.map.find_cell(seeker_pos)
        if not start_cell:
            return None

        # Attempt to go towards open space if stuck
        if self._stuck_counter >= 4 and self._stuck_reset_cooldown == 0:
            escape = self._escape_move(start_cell, hider_pos, seeker_pos)
            if escape:
                self._stuck_counter = 0
                self._stuck_reset_cooldown = 10
                self._last_move = escape
                return escape
        if self._stuck_reset_cooldown > 0:
            self._stuck_reset_cooldown -= 1

        # Periodically pick a new target cell
        recalc = (
            self._target_cell is None
            or self._tick - self._last_target_tick >= self._target_recalc_interval
            or (self._last_seeker_cell is not None and seeker_cell != self._last_seeker_cell)
        )
        if recalc:
            self._target_cell = self._pick_target_cell(seeker_cell, start_cell, seeker_pos)
            self._last_target_tick = self._tick
            self._last_seeker_cell = seeker_cell

        # Fallback if no target is available
        target_cell = self._target_cell
        if not target_cell:
            move = self._flee_step(hider_pos, seeker_pos)
            self._last_move = move
            return move

        target_point = self._cell_centroid[target_cell]

        # Go straight to target if in line of sight
        if self.map.has_line_of_sight(hider_pos, target_point):
            self._cell_path = None
            self._waypoints = None
            move = self._move_toward(hider_pos, target_point)
            move = self._keep_inside(hider_pos, move, margin=0.25)
            self._last_move = move
            return move

        # Build or reuse A* path to the target cell
        need_new_path = (
            self._cell_path is None
            or self._path_target != target_cell
            or not self._cell_path
            or start_cell not in self._cell_path
        )

        if need_new_path:
            path = astar(start_cell, target_cell)
            if not path:
                self._cell_path = None
                self._waypoints = None
                move = self._flee_step(hider_pos, seeker_pos)
                self._last_move = move
                return move
            self._cell_path = path
            self._path_target = target_cell
            self._waypoints = None
        else:
            while (
                self._cell_path
                and len(self._cell_path) > 1
                and self._cell_path[1].polygon.covers(hider_pos)
            ):
                self._cell_path.pop(0)

        if not self._cell_path:
            move = self._flee_step(hider_pos, seeker_pos)
            self._last_move = move
            return move

        # Build portal waypoints (take shortcut if in line of sight)
        if not self._waypoints:
            portal_points: list[shapely.Point] = []
            for idx in range(len(self._cell_path) - 1):
                next_cell = self._cell_path[idx + 1]
                border = self._cell_path[idx].neighbors.get(next_cell)
                if border:
                    try:
                        portal_points.append(border.interpolate(0.5, normalized=True))
                        continue
                    except ValueError:
                        pass
                portal_points.append(self._cell_centroid[next_cell])
            portal_points.append(target_point)

            self._waypoints = []
            anchor = hider_pos
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
            move = self._flee_step(hider_pos, seeker_pos)
            self._last_move = move
            return move

        # Trim waypoints that are close
        clear_distance = max(1.0, self.max_speed * 0.5)
        while (
            len(self._waypoints) > 1
            and hider_pos.distance(self._waypoints[0]) <= clear_distance
        ):
            self._waypoints.pop(0)

        if not self._waypoints:
            move = self._flee_step(hider_pos, seeker_pos)
            self._last_move = move
            return move

        # Choose immediate waypoint
        waypoint = self._waypoints[0]
        if (
            len(self._waypoints) > 1
            and hider_pos.distance(waypoint) <= clear_distance
            and self.map.has_line_of_sight(hider_pos, self._waypoints[1])
        ):
            waypoint = self._waypoints[1]

        # Final movement step with logic to ensure that hider is in-bounds and can escape
        move = self._move_toward(hider_pos, waypoint)
        move = self._keep_inside(hider_pos, move, margin=0.25)
        if move is None:
            move = self._escape_move(start_cell, hider_pos, seeker_pos)
        if move is None:
            move = self._last_move
        self._last_move = move
        return move

    @property
    def is_seeker(self) -> bool:
        return False

    def _pick_target_cell(
        self, seeker_cell: Optional[NavMeshCell], start_cell: NavMeshCell, seeker_pos: shapely.Point
    ) -> Optional[NavMeshCell]:
        # score all cells, deduct from score if in LOS for the top few
        reference = seeker_cell if seeker_cell else start_cell
        current_dist = start_cell.distance(reference)
        candidates: list[tuple[float, NavMeshCell]] = []
        for cell in self._cells:
            distance_score = cell.distance(reference)
            branch_score = self._cell_branch[cell] * 4.0
            edge_bonus = self._cell_boundary_dist[cell] * 1.0
            speed_bonus = self._cell_branch[cell] * (self.max_speed * 0.5)
            closer_penalty = (current_dist - distance_score) * 3.0 if distance_score < current_dist else 0.0
            base = distance_score + branch_score + edge_bonus + speed_bonus - closer_penalty
            candidates.append((base, cell))

        candidates.sort(reverse=True, key=lambda x: x[0])
        top = candidates[:20] if len(candidates) > 20 else candidates

        best_cell = None
        best_score = -1e9
        head_start_bonus = 1.6 if not self._seeker_started else 1.0
        for base, cell in top:
            score = head_start_bonus * base
            if seeker_cell and self.map.has_line_of_sight(seeker_pos, self._cell_centroid[cell]):
                score -= cell.distance(reference) * 3.0
            if score > best_score:
                best_score = score
                best_cell = cell
        return best_cell

    def _escape_move(
        self, start_cell: NavMeshCell, hider_pos: shapely.Point, seeker_pos: shapely.Point
    ) -> Optional[tuple[float, float]]:
        # Try moving toward the widest portal first, then fall back to flee or map center
        best_move = None
        best_width = -1.0
        for neighbor, border in start_cell.neighbors.items():
            width = border.length
            target = self._cell_centroid.get(neighbor, neighbor.polygon.centroid)
            move = self._move_toward(hider_pos, target)
            move = self._keep_inside(hider_pos, move, margin=0.25)
            if move and width > best_width:
                best_width = width
                best_move = move
        if best_move:
            return best_move
        # Fallbacks
        move = self._keep_inside(hider_pos, self._flee_step(hider_pos, seeker_pos), margin=0.25)
        if move:
            return move
        return self._keep_inside(hider_pos, self._move_toward(hider_pos, self._map_centroid), margin=0.25)

    def _move_toward(self, start: shapely.Point, target: shapely.Point) -> tuple[float, float] | None:
        # Scale movement to max_speed
        dx = target.x - start.x
        dy = target.y - start.y
        dist = math.hypot(dx, dy)
        if dist == 0:
            return None
        speed = min(dist, self.max_speed)
        scale = speed / dist
        return (dx * scale, dy * scale)

    def _keep_inside(
        self, start: shapely.Point, move: Optional[tuple[float, float]], margin: float = 0.0
    ) -> Optional[tuple[float, float]]:
        # Clamp movement within a shrunken polygon to keep distance from walls (avoiding getting stuck)
        if move is None:
            return None
        dx, dy = move
        end_point = shapely.Point(start.x + dx, start.y + dy)
        line = shapely.LineString([start, end_point])
        padded = self._safe_poly(margin)
        if padded.covers(line):
            return move
        for factor in (0.9, 0.75, 0.6, 0.45, 0.3, 0.15):
            scaled = (dx * factor, dy * factor)
            test_point = shapely.Point(start.x + scaled[0], start.y + scaled[1])
            test_line = shapely.LineString([start, test_point])
            if padded.covers(test_line):
                return scaled
        # Perpendicular slide
        perp_dx, perp_dy = -dy, dx
        perp_len = math.hypot(perp_dx, perp_dy) or 1.0
        perp_scale = min(self.max_speed, math.hypot(dx, dy)) / perp_len * 0.5
        slide = (perp_dx * perp_scale, perp_dy * perp_scale)
        if padded.covers(shapely.LineString([start, shapely.Point(start.x + slide[0], start.y + slide[1])])):
            return slide
        slide = (-slide[0], -slide[1])
        if padded.covers(shapely.LineString([start, shapely.Point(start.x + slide[0], start.y + slide[1])])):
            return slide
        return (dx * 0.1, dy * 0.1)

    def _safe_poly(self, margin: float) -> shapely.Polygon:
        # Cache the polygons to avoid repeated buffer() calls
        if margin <= 0:
            return self.map.polygon
        cached = self._safe_polys.get(margin)
        if cached is not None:
            return cached
        padded = self.map.polygon.buffer(-margin)
        if padded.is_empty:
            padded = self.map.polygon
        self._safe_polys[margin] = padded
        return padded

    def _flee_step(self, hider_pos: shapely.Point, seeker_pos: shapely.Point) -> tuple[float, float] | None:
        # Step away from the seeker
        dx = hider_pos.x - seeker_pos.x
        dy = hider_pos.y - seeker_pos.y
        dist = math.hypot(dx, dy)
        if dist == 0:
            return None
        scale = min(dist, self.max_speed) / dist
        return (dx * scale, dy * scale)
