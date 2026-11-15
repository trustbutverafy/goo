"""The sampler's motif: a 64 x 64 pixel-art plate of flat colours and one-texel features, generated here.

  top left      cream, with a navy block, a red diagonal line and a row of navy glyphs
  top right     near-black beside the cream, crossed by a yellow horizontal and vertical line, a pale blue
                diagonal the other way, and a small cream face (two eyes, a mouth)
  bottom left   terracotta bricks in pale mortar (a one-texel network), a two-green dither (checker)
                beside an ochre field with brown glyphs
  bottom right  grey-blue, with a red vertical line, a navy diagonal and a two-texel navy bar

The darkest colour is reachable with goo's stock pigments (carbon bottoms out near a reflectance of 0.015).
Returns sRGB (uint8) and object ids, at one texel per pixel.
"""
import numpy as np

C = dict(cream=(236, 226, 200), black=(40, 40, 46), navy=(36, 52, 96), red=(196, 48, 40), yellow=(232, 196, 64),
         pale=(150, 190, 220), brick=(170, 84, 56), mortar=(214, 200, 176), green1=(96, 148, 76), green2=(52, 96, 48),
         ochre=(210, 170, 90), brown=(80, 50, 30), grey=(120, 136, 160))

GLYPHS = {   # 3 x 5, one-texel strokes
    "plus": ["...", ".#.", "###", ".#.", "..."], "ell": ["#..", "#..", "#..", "#..", "###"],
    "tee": ["###", ".#.", ".#.", ".#.", ".#."], "ring": [".#.", "#.#", "#.#", "#.#", ".#."],
    "ex": ["#.#", "#.#", ".#.", "#.#", "#.#"], "zed": ["###", "..#", ".#.", "#..", "###"],
}


def plate(n=64):
    assert n == 64, "the layout is drawn for 64 x 64 texels"
    img = np.zeros((n, n, 3), np.uint8); ids = np.zeros((n, n), int)
    def rect(x0, y0, x1, y1, col, oid=None):
        img[y0:y1, x0:x1] = C[col]
        if oid is not None: ids[y0:y1, x0:x1] = oid
    def glyph(name, x, y, col):
        for j, row in enumerate(GLYPHS[name]):
            for i, ch in enumerate(row):
                if ch == "#": img[y + j, x + i] = C[col]
    rect(0, 0, 34, 28, "cream", 0); rect(34, 0, 64, 28, "black", 1)
    rect(0, 28, 32, 48, "brick", 2); rect(0, 48, 16, 64, "green1", 3); rect(16, 48, 32, 64, "ochre", 4); rect(32, 28, 64, 64, "grey", 5)
    rect(4, 4, 13, 12, "navy", 6)                                            # a flat block of its own
    for i in range(14): img[1 + i, 15 + i] = C["red"]                         # diagonal, down to the right
    for k, g in enumerate(["plus", "ell", "tee", "ring", "ex", "zed"]): glyph(g, 3 + 5 * k, 19, "navy")
    img[8, 36:62] = C["yellow"]; img[2:26, 48] = C["yellow"]                  # crossing lines
    for i in range(10): img[25 - i, 36 + i] = C["pale"]                       # diagonal, up to the right
    img[16, 54] = img[16, 58] = C["cream"]                                   # a face: eyes, mouth
    for x, y in ((53, 19), (54, 20), (55, 20), (56, 20), (57, 20), (58, 19)): img[y, x] = C["cream"]
    for y in range(31, 48, 4): img[y, 0:32] = C["mortar"]                     # bricks: courses ...
    for row, y0 in enumerate(range(28, 48, 4)):                              # ... and staggered joints
        for x in range(2 + 3 * (row % 2), 32, 6): img[y0:y0 + 3, x] = C["mortar"]
    yy, xx = np.mgrid[48:64, 0:16]; img[48:64, 0:16][(yy + xx) % 2 == 0] = C["green2"]   # dither
    glyph("ring", 19, 51, "brown"); glyph("ex", 24, 51, "brown"); glyph("plus", 19, 57, "brown"); glyph("ell", 25, 57, "brown")
    img[30:62, 40] = C["red"]                                                 # vertical line
    for i in range(12): img[30 + i, 44 + i] = C["navy"]                       # diagonal
    img[50:52, 44:62] = C["navy"]                                             # a two-texel bar
    return img, ids
