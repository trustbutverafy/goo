"""goo.ports: bodies and reservoirs coupled to the world (SPEC §8). docs/PORTS.md.

    Paint, Vessel                     recipes and the bodies that hold them
    Bench                             a World with its bodies; every act a method returning a goo.ledger.Receipt:
                                      paint, mix, fill, pour, flood, drop, drip, spray, tape, peel, tilt, level, air,
                                      wait, hold, release, settle
    geometry                          regions, paths and footprints as plain data
    grounds                           grounds by name (data with provenance) and make_ground
"""
from goo.ports.paint import Paint, Vessel, PortError
from goo.ports.bench import Bench, Tape, RESERVOIRS, WALL_NOTE
from goo.ports import geometry
from goo.ports.geometry import GeometryError
from goo.ports.grounds import GROUNDS, GroundSpec, make_ground

__all__ = ["Paint", "Vessel", "PortError", "Bench", "Tape", "RESERVOIRS", "WALL_NOTE", "geometry", "GeometryError",
           "GROUNDS", "GroundSpec", "make_ground"]
