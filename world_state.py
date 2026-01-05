from dataclasses import dataclass

import shapely


@dataclass(frozen=True)
class WorldState:

    hider_position: shapely.Point | None
    seeker_position: shapely.Point
