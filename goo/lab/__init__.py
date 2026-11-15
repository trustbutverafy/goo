"""goo.lab: what a substance would look like as paint, and how it is known (SPEC §10). docs/LAB.md.

Imports goo.substance, goo.film, goo.light and goo.observe (and units, evidence, ledger); never goo.world or goo.ports:
the lab works without the painting engine. Films are made by composition, not by flow.

    recipes      binders, paints, grounds, the room: recipes built from library substances
    dry          the dry film: reactions to completion, volatiles to the environment's activity
    drawdown     Card: masstone and tints, over black and white, wet and dry, per binder and illuminant, gloss, IR
    sheet        the phenotype: hiding, tinting strength, drying shift, gloss, infrared response
    admissible   the laws passed and failed with shortfalls; what keeps a substance from real
    sensitivity  every number again with size halved and doubled and index perturbed
    search       nearest possible substance to a target, and the impossible one that hits it
    contribute   one JSON file per substance; validate
    render       card PNGs and the contact sheet
"""
from goo.lab.recipes import Binder, Paint, OIL, WATERCOLOUR, ACRYLIC, WATER, BINDERS, ROOM, LAB_BANDS
from goo.lab.dry import dry, Dried
from goo.lab.drawdown import drawdown, Card
from goo.lab.sheet import sheet
from goo.lab.admissible import admissible
from goo.lab.sensitivity import sensitivity, variants
from goo.lab.contribute import to_json, from_json, validate, write
from goo.lab.render import render_card, contact_sheet

__all__ = ["Binder", "Paint", "OIL", "WATERCOLOUR", "ACRYLIC", "WATER", "BINDERS", "ROOM", "LAB_BANDS", "dry", "Dried",
           "drawdown", "Card", "sheet", "admissible", "sensitivity", "variants", "to_json", "from_json", "validate",
           "write", "render_card", "contact_sheet"]
