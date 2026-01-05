from copy import deepcopy
from dataclasses import dataclass
from functools import cache
from typing import Self
import pygame
import shapely
from shapely import constrained_delaunay_triangles
import numpy as np


class NavMeshCell:

    count = 0

    def __init__(self, polygon: shapely.Polygon):
        self._polygon = polygon
        self._neighbors: dict[Self, shapely.LineString] = {}
        self._id = NavMeshCell.count
        NavMeshCell.count += 1

    def __str__(self) -> str:
        return f"Cell_{self._id}"

    def __repr__(self) -> str:
        return str(self)

    @property
    def neighbors(self) -> dict[Self, shapely.LineString]:
        return self._neighbors

    def add_neighbor(self, other: Self):
        border = self.polygon & other.polygon
        if not isinstance(border, shapely.LineString):
            return
        self.neighbors[other] = border
        other.neighbors[self] = border

    def distance(self, other: Self) -> float:
        return shapely.distance(self.polygon.centroid, other.polygon.centroid)

    @property
    def polygon(self) -> shapely.Polygon:
        return self._polygon

    @property
    def id(self) -> int:
        return self._id


class NavMesh:

    pygame.font.init()
    font = pygame.font.Font(None, 20)
    rng = np.random.default_rng()

    def __init__(self, polygon: shapely.Polygon) -> None:
        self._polygon = polygon
        self._cells: set[NavMeshCell] = set()
        triangles = constrained_delaunay_triangles(polygon)
        for triangle in shapely.get_parts(triangles):
            self.add_cell(triangle)

    @property
    def polygon(self) -> shapely.Polygon:
        return self._polygon

    @property
    def cells(self) -> set[NavMeshCell]:
        return self._cells

    def add_cell(self, shape: shapely.Polygon):
        cell = NavMeshCell(shape)
        for other in self.cells:
            other.add_neighbor(cell)
        self.cells.add(cell)

    def find_cell(self, coord: shapely.Point) -> NavMeshCell | None:
        for cell in self.cells:
            if cell._polygon.intersects(coord):
                return cell
        return None

    @cache
    def render(
        self, screen: pygame.Surface, draw_cells: bool = False
    ) -> pygame.Surface:
        output = pygame.Surface(screen.get_size())
        for poly in shapely.get_parts(self.polygon):
            pygame.draw.polygon(output, "#ffffff", poly.exterior.coords)
            for hole in poly.interiors:
                pygame.draw.polygon(output, "#000000", hole.coords)
        if not draw_cells:
            return output
        for cell in self.cells:
            pygame.draw.polygon(output, "#ff0000", cell._polygon.exterior.coords, 2)  # type: ignore
        return output

    def in_bounds(self, coord: shapely.Point) -> bool:
        return self.polygon.contains(coord)

    def random_position(self) -> shapely.Point:
        min_x, min_y, max_x, max_y = self.polygon.bounds
        while True:
            point = shapely.Point(NavMesh.rng.uniform([min_x, min_y], [max_x, max_y]))
            if self.polygon.contains(point):
                return point

    def has_line_of_sight(self, point_a: shapely.Point, point_b: shapely.Point) -> bool:
        line = shapely.LineString([point_a, point_b])
        return self.polygon.contains(line)
