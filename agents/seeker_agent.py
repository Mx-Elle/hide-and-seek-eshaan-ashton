from agent_base import Agent
from world_state import WorldState
from Mesh.nav_mesh import NavMesh


class DumbSeeker(Agent):

    def __init__(self, world_map: NavMesh, max_speed: float):
        Agent.__init__(self, world_map, max_speed)
        self.name = "Dumb Seeker"

    def act(self, state: WorldState) -> tuple[float, float] | None:
        # TODO: Write me!
        ...

    @property
    def is_seeker(self) -> bool:
        return True
