from pathlib import Path
import pickle
import sys
import numpy as np
import shapely

path_to_src = Path(__file__).parent.parent
sys.path.insert(0, str(path_to_src))

from Mesh.nav_mesh import NavMesh

RNG = np.random.default_rng()


def scale_to_fit(
    map_poly: shapely.Polygon, width: float, height: float
) -> shapely.Polygon:
    minx, miny, maxx, maxy = map_poly.bounds
    map_poly = shapely.affinity.translate(map_poly, -minx, -miny)
    map_poly = shapely.affinity.scale(
        map_poly, width / (maxx - minx), height / (maxy - miny), origin=(0, 0)
    )
    return map_poly


def rc_to_index(row: int, col: int, size: tuple[int, int]) -> int:
    return row * size[1] + col


def index_to_rc(idx: int, size: tuple[int, int]) -> tuple[int, int]:
    return (idx // size[1], idx % size[1])


def _grid_neighbors(size: tuple[int, int]) -> np.ndarray:
    rows, cols = size
    neighbors = np.zeros((rows * cols, rows * cols))
    for row in range(rows):
        for col in range(cols):
            idx = rc_to_index(row, col, size)
            for neighbor in [
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ]:
                if not (0 <= neighbor[0] < size[0] and 0 <= neighbor[1] < size[1]):
                    continue
                n_idx = rc_to_index(*neighbor, size)
                neighbors[idx, n_idx] = 1
                neighbors[n_idx, idx] = 1
    return neighbors


def make_maze(
    width: int,
    height: int,
    size: tuple[int, int],
    corridor_width: float = 0.7,
) -> NavMesh:
    rows, cols = size
    neighbors = _grid_neighbors(size)
    connections = np.zeros((rows * cols, rows * cols))
    closed = set()
    frontier = [
        rc_to_index(RNG.integers(rows, dtype=int), RNG.integers(cols, dtype=int), size)
    ]
    while frontier:
        cell = frontier[-1]
        if RNG.uniform(0, 1) < 0.2:
            my_neighbors = [n for n in np.where(neighbors[cell])[0]]
        else:
            my_neighbors = [
                n
                for n in np.where(neighbors[cell])[0]
                if n not in closed and n not in frontier
            ]
        if len(my_neighbors) == 0:
            closed.add(cell)
            frontier.remove(cell)
            continue
        next_cell = RNG.choice(my_neighbors)
        connections[cell, next_cell] = 1
        frontier.append(next_cell)
    points = {
        rc_to_index(r, c, size): shapely.Point(r, c)
        for r in range(rows)
        for c in range(cols)
    }
    links = list(zip(*np.where(connections)))
    lines = []
    for a, b in links:
        lines.append(shapely.LineString([points[a], points[b]]))
    poly = shapely.union_all(lines).buffer(
        0.5 * corridor_width, cap_style="square", quad_segs=1
    )
    poly = shapely.simplify(poly, 0.01)
    poly = scale_to_fit(poly, width, height)
    return NavMesh(poly)


def random_point_in_polygon(
    polygon: shapely.Polygon, corridor_width: float
) -> shapely.Point:
    minx, miny, maxx, maxy = polygon.bounds
    margin = corridor_width / 2
    while True:
        x, y = RNG.uniform(minx + margin, maxx - margin), RNG.uniform(
            miny + margin, maxy - margin
        )
        p = shapely.Point(x, y)
        if polygon.contains(p):
            return p


def tunnel_between(
    room_a: shapely.Polygon, room_b: shapely.Polygon, tunnel_width: float
) -> shapely.Polygon:
    pa = random_point_in_polygon(room_a, tunnel_width)
    pb = random_point_in_polygon(room_b, tunnel_width)
    if RNG.uniform(0, 1) < 0.5:  # horizontal then vertical
        line = shapely.LineString([pa, (pb.x, pa.y), pb])
    else:
        line = shapely.LineString([pa, (pa.x, pb.y), pb])
    return line.buffer(
        tunnel_width / 2, cap_style="square", join_style="mitre", quad_segs=0
    )


def make_dungeon(
    width: int,
    height: int,
    min_room_size: float,
    max_room_size: float,
    max_rooms: int,
    extra_corridors: int,
    tunnel_width: int,
) -> NavMesh:
    rooms = []
    corridors = []
    for _ in range(max_rooms):
        room_width = RNG.uniform(min_room_size, max_room_size)
        room_height = RNG.uniform(min_room_size, max_room_size)
        x = RNG.uniform(0, width - room_width)
        y = RNG.uniform(0, height - room_height)
        room = shapely.Polygon(
            [
                (x, y),
                (x + room_width, y),
                (x + room_width, y + room_height),
                (x, y + room_height),
            ]
        )
        if room.overlaps(shapely.union_all(rooms)):
            continue
        if len(rooms) > 0 and not room.overlaps(shapely.union_all(corridors)):
            connection = RNG.choice(rooms)
            corridor = tunnel_between(room, connection, tunnel_width)
            corridors.append(corridor)
        rooms.append(room)
    for _ in range(extra_corridors):
        room_a = RNG.choice(rooms)
        room_b = RNG.choice([r for r in rooms if r != room_a])
        corridors.append(tunnel_between(room_a, room_b, tunnel_width))
    poly = shapely.union_all(rooms + corridors)
    minx, miny, _, _ = poly.bounds
    poly = shapely.affinity.translate(poly, -minx, -miny)
    return NavMesh(poly)  # type: ignore


def make_spaghetti(
    width: int, height: int, n_corridors: int, min_width: float, max_width: float
) -> NavMesh:
    poly = shapely.Polygon()
    n = 0
    while n < n_corridors:
        start_x, end_x, start_y, end_y = RNG.uniform(0, [width, width, height, height])
        hall_width = RNG.uniform(min_width, max_width)
        hall = shapely.LineString([(start_x, start_y), (end_x, end_y)]).buffer(
            hall_width, cap_style="square"
        )
        if n == 0 or hall.intersects(poly):
            n += 1
            poly |= hall
    minx, miny, _, _ = poly.bounds
    poly = shapely.affinity.translate(poly, -minx, -miny)
    return NavMesh(poly)  # type: ignore


def main():
    parent_dir = Path.cwd()
    for i in range(10, 21, 5):
        print(f"Generating spaghetti {i}")
        maze = make_spaghetti(800, 800, i, 5, 20)
        with open(parent_dir / f"maps/spaghetti_{i}.pkl", "wb") as f:
            pickle.dump(maze, f)
    for i in range(5, 31, 5):
        print(f"Generating maze {i}")
        maze = make_maze(800, 800, (i, i))
        with open(parent_dir / f"maps/maze_{i}.pkl", "wb") as f:
            pickle.dump(maze, f)
    for i in range(5, 55, 10):
        print(f"Generating dungeon {i}")
        maze = make_dungeon(800, 800, 30, 100, i, 0, 10)
        with open(parent_dir / f"maps/dungeon_{i}.pkl", "wb") as f:
            pickle.dump(maze, f)


if __name__ == "__main__":
    main()
