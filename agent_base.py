from abc import ABC
import math
from threading import Lock, Thread
from Mesh.nav_mesh import NavMesh
from world_state import WorldState


class Agent(ABC, Thread):

    def __init__(self, world_map: NavMesh, max_speed: float):
        Thread.__init__(self)
        self.name = "Replace this with your bot's name"
        self.map = world_map
        self.max_speed = max_speed
        self._state: WorldState | None = None
        self._lock = Lock()
        self._next_move: tuple[float, float] | None = None

    def run(self) -> None:
        while True:
            if not self._state:
                continue
            with self._lock:
                state = self._state
            move = self.act(state)
            with self._lock:
                self._next_move = move

    def act(self, state: WorldState) -> tuple[float, float] | None:
        # TODO: Write me!
        ...

    @property
    def is_seeker(self) -> bool:
        # TODO: This should just return True or False
        ...
