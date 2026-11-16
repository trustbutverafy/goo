"""Nocturne: the far shore. One act list in metres, so the same painting can be laid on a coarse grid (a
cartoon) or a fine one. mode 'dry': films laid already dry under a held clock (a study); mode 'wet': real paint."""
import warnings; warnings.simplefilter("ignore")
import sys, json, time, numpy as np
from goo.units import um, mm
from goo.evidence import Provenance as P
from goo.substance import library
from goo.substance.description import Substance
from goo.substance.dielectric import Oscillators, Gauss, eV
from goo.substance.morphology import Sphere, LogNormal

def register():
    if "lamp" not in library.BY_NAME:
        lamp = Substance("lamp", Oscillators(2.25, (Gauss(-0.006, 2.10 * eV, 0.28 * eV),)),
                         Sphere(LogNormal(0.3 * um, 1.4)), 2000.0, reservoir=1e5,
                         prov=dict(eps=P("guessed", note="a causal band of negative strength at 590 nm: gain. "
                                                         "Impossible: fails passivity."),
                                   morph=P("guessed"), rho=P("guessed"), reservoir=P("guessed")))
        library.BY_NAME["lamp"] = lamp                      # a painter's own substance, registered for this session

SUBS = ["water", "gum_arabic", "rutile", "soot", "ultramarine", "rose", "cadmium_sulfide", "hematite", "green",
        "dioxazine", "lamp"]
PARTS = dict(                                   # pigment and binder by parts of volume, dry
    priming={"rutile": 4, "gum_arabic": 1},
    haze={"rutile": .25, "cadmium_sulfide": .04, "hematite": .015, "soot": .015, "gum_arabic": 3},
    blue_grey={"ultramarine": 1, "soot": .05, "hematite": .03, "rutile": .12, "gum_arabic": 3},
    deep={"ultramarine": 1, "soot": .10, "gum_arabic": 2.5},
    mauve={"ultramarine": .6, "rose": .12, "hematite": .1, "rutile": .2, "soot": .03, "gum_arabic": 3},
    glow={"cadmium_sulfide": 1, "hematite": .06, "rutile": .3, "gum_arabic": 5},
    ember={"cadmium_sulfide": 1, "hematite": .25, "gum_arabic": 4},
    water_deep={"ultramarine": 1, "green": .5, "soot": .2, "gum_arabic": 3},
    land={"soot": 1, "hematite": .3, "ultramarine": .5, "gum_arabic": 2},
    slate={"dioxazine": .3, "soot": .08, "rutile": .15, "hematite": .05, "gum_arabic": 3},
    mist={"rutile": .12, "gum_arabic": 3},
    white={"rutile": 1, "gum_arabic": 1},
    lamp={"lamp": 1, "gum_arabic": 2},
)
WM, HM = 409.6 * mm, 307.2 * mm                 # the panel: 16 x 12 inches, near enough

def recipe(name, water=0.0):
    p = PARTS[name]; tot = sum(p.values())
    r = {k: (1 - water) * v / tot for k, v in p.items()}
    if water: r["water"] = water
    return r

class Plan:
    def __init__(self, min_r=0.0, rng=7):
        self.acts, self.min_r, self.rng = [], min_r, np.random.default_rng(rng)
        self.seed = 100
    def xy(self, u, v): return [float(u * WM), float(v * HM)]
    def add(self, **a): self.acts.append(a)
    def group(self, name): self.acts.append({"group": name})
    def spray(self, paint, pts, R, t, wet=1.0):
        """t: dried thickness (m) meant near the band's centre line, roughly; R: cone radius (m)."""
        smooth = isinstance(pts, dict)
        pts = [self.xy(*p) for p in (pts["smooth"] if smooth else pts)]
        L = sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])) or 2 * R
        vol = t * L * 2 * R * 0.6 / wet
        self.seed += 1
        if len(pts) == 1: pts = [pts[0], [pts[0][0] + 1e-4, pts[0][1]]]
        path = {"points": pts, "units": "m"}
        if smooth: path["smooth"] = 3
        self.add(act="spray", vessel=paint, path=path, flow=f"{vol / 10.0 * 1e9:.6g} uL/s", duration="10 s",
                 cone_radius=f"{R / mm:.4g} mm", droplet_size="40 um", seed=self.seed, edge=f"{0.5 * R / mm:.4g} mm")
    def band(self, paint, v, R, t, u0=-0.04, u1=1.04):
        self.spray(paint, [(u0, v), (u1, v)], R * HM, t * um)
    def flood(self, paint, region, t, edge=0.0):
        a = dict(act="flood", vessel=paint, region=dict(region, units="m"), thickness=f"{t:g} um")
        if edge: a["edge"] = f"{edge / mm:.4g} mm"
        self.add(**a)
    def disc(self, u, v, r): return {"disc": self.xy(u, v) + [float(max(r, self.min_r))]}
    def rect(self, u0, v0, u1, v1):
        x0, y0 = self.xy(u0, v0); x1, y1 = self.xy(u1, v1)
        if x1 - x0 < 2 * self.min_r: c = (x0 + x1) / 2; x0, x1 = c - self.min_r, c + self.min_r
        if y1 - y0 < 2 * self.min_r: c = (y0 + y1) / 2; y0, y1 = c - self.min_r, c + self.min_r
        return {"rect": [x0, y0, x1, y1]}
    def poly(self, pts): return {"polygon": [self.xy(*p) for p in pts]}

HORIZON = 0.42
D = HORIZON - 0.47
SKYLINE = [(u, v + D) for u, v in [(0, .47), (0, .445), (.06, .44), (.12, .448), (.20, .438), (.27, .45), (.33, .446),
           (.40, .452), (.47, .452), (.47, .435), (.50, .435), (.50, .442), (.53, .442), (.53, .428), (.56, .428),
           (.56, .44), (.595, .44), (.595, .415), (.605, .385), (.615, .415), (.615, .438), (.66, .438), (.66, .43),
           (.69, .43), (.72, .42), (.78, .39), (.85, .365), (.92, .355), (1.0, .36), (1.0, .47)]]
LAMPS = [(.075, 0.9), (.15, 0.7), (.305, 0.8), (.355, 1.0), (.44, 0.7), (.515, 0.9), (.575, 1.1), (.64, 0.8), (.775, 1.0), (.90, 0.7)]
LAMP_V = HORIZON - .008
WINDOWS = [(.485, .444 + D), (.545, .436 + D), (.675, .436 + D)]
MOON = (.22, .13, 8.0 * mm)
BOAT_LAMP = (.701, .615, 1.0 * mm)

def compose(min_r=0.0, stars=False):
    p = Plan(min_r); rng = p.rng; h = HORIZON
    p.group("priming")
    p.flood("priming", {"rect": [0, 0, WM, HM]}, 40)
    p.group("reserves")
    p.add(act="tape", name="moon", region=dict(p.disc(*MOON), units="m"))
    for i, (u, r) in enumerate(LAMPS):
        p.add(act="tape", name=f"lamp{i}", region=dict(p.disc(u, LAMP_V, r * mm), units="m"))
    for i, (u, v) in enumerate(WINDOWS):
        p.add(act="tape", name=f"win{i}", region=dict(p.rect(u - .0016, v - .003, u + .0016, v + .003), units="m"))
    p.add(act="tape", name="boatlamp", region=dict(p.disc(*BOAT_LAMP), units="m"))
    p.add(act="tape", name="cabin", region=dict(p.rect(.700, .781, .712, .791), units="m"))
    p.group("tone")
    p.flood("haze", {"rect": [0, 0, WM, HM]}, 2.5)
    p.group("sky")
    p.band("glow", h - .03, .05, 1.0, .1, 1.0)
    p.band("glow", h - .014, .03, 3, .28, .98); p.band("glow", h - .012, .016, 3, .44, .80)
    p.band("mauve", h - .075, .07, 2)
    p.band("blue_grey", h - .15, .09, 2.2); p.band("blue_grey", h - .23, .09, 2.6)
    p.band("deep", h - .31, .09, 2.5); p.band("deep", h - .39, .10, 3); p.band("deep", -.02, .10, 3)
    p.group("water")
    p.band("land", h + .020, .028, 2.2)
    p.band("mauve", h + .078, .04, 0.9)
    p.band("mauve", h + .13, .06, 2)
    p.band("blue_grey", h + .21, .09, 2.0); p.band("blue_grey", h + .29, .09, 1.5)
    p.band("water_deep", h + .36, .09, 1.5); p.band("water_deep", h + .44, .09, 2.0)
    p.band("water_deep", h + .52, .09, 2.0); p.band("water_deep", 1.02, .10, 2.0)
    p.group("clouds")
    p.spray("slate", {"smooth": [(.26, h - .084), (.36, h - .094), (.47, h - .089), (.58, h - .097), (.68, h - .090)]}, .010 * HM, 2.5 * um)
    p.spray("slate", {"smooth": [(.48, h - .050), (.60, h - .058), (.72, h - .054), (.84, h - .062), (.93, h - .057)]}, .008 * HM, 2.5 * um)
    p.group("land")
    p.flood("slate", p.poly([(0, h), (0, h - .05), (.08, h - .058), (.18, h - .045), (.28, h - .052), (.38, h - .038), (.46, h - .03), (.46, h)]), 3)
    p.flood("slate", p.poly([(.62, h), (.62, h - .03), (.70, h - .065), (.80, h - .085), (.90, h - .075), (1.0, h - .095), (1.0, h)]), 5)
    p.flood("land", p.poly(SKYLINE), 12)
    p.spray("land", [(.74, h + .05), (1.04, h + .058)], .05 * HM, 2.6 * um)
    p.flood("land", p.poly([(.60, .80), (.80, .80), (.785, .835), (.62, .835)]), 12)
    p.flood("land", p.rect(.68, .775, .74, .80), 12)
    p.flood("land", p.rect(.6995, .62, .7025, .775), 12)
    p.spray("land", [(.63, .845), (.77, .845)], .007 * HM, 2.5 * um)
    p.group("lights")
    p.band("mist", h + .016, .010, 0.4)
    p.spray("mist", [(MOON[0], MOON[1])], 20 * mm, 1.0 * um)
    for u, r in LAMPS:
        p.spray("ember", [(u, LAMP_V)], 2.6 * mm, 1.5 * um)
    def dash(paint, u, v, ln, R, t):
        p.spray(paint, [(u - ln / WM / 2, v), (u + ln / WM / 2, v)], R, t * um)
    for k in range(80):                                    # the moon's road on the water
        f = rng.random() ** 0.7; v = h + .03 + (.97 - h - .03) * f
        u = MOON[0] + rng.normal(0, .006 + .045 * f)
        core = abs(u - MOON[0]) < .012 + .03 * f
        dash("white" if core else "mist", u, v, (1.5 + 9 * rng.random()) * mm * (0.4 + 1.6 * f), (0.35 + 0.3 * rng.random()) * mm * (0.5 + 1.5 * f), 4 if core else 3)
    for k in range(34):                                    # ripples: small and far near the shore, long and near below
        f = rng.random(); v = h + .05 + (.97 - h - .05) * f
        u = MOON[0] + rng.normal(0, .22) if k % 2 else rng.random()
        dash("mist", u, v, (3 + 18 * rng.random()) * mm * (0.3 + 1.7 * f), (0.25 + 0.25 * rng.random()) * mm * (0.5 + 1.5 * f), 0.7 + 1.8 * rng.random())
    for k in range(9):                                     # moonlit water behind the boat, so the hull reads
        dash("mist", .70 + rng.normal(0, .07), .775 + .05 * rng.random(), (6 + 14 * rng.random()) * mm, 0.45 * mm, 1.2 + rng.random())
    for k in range(4):                                     # the cabin window on the water
        dash("ember", .706 + rng.normal(0, .002), .852 + .012 * k, (1.5 + k) * mm, 0.4 * mm, 4 - 0.7 * k)
    for (u, r) in LAMPS + [(BOAT_LAMP[0], 1.0)]:           # lamp reflections: broken, wavering, fading downward
        boat = u == BOAT_LAMP[0]
        v0 = .845 if boat else h + .012
        n = 5 if boat else int(7 + 6 * rng.random())
        for k in range(n):
            f = k / n; v = v0 + (.05 if boat else .13) * f + .004 * rng.random()
            dash("ember" if k % 3 else "glow", u + rng.normal(0, .0015 + .003 * f), v, (1.2 + 2.2 * f + rng.random()) * mm, (0.3 + 0.25 * f) * mm, 5 * (1 - 0.6 * f))
    p.group("peel")
    for n in ["moon", "boatlamp", "cabin"] + [f"lamp{i}" for i in range(len(LAMPS))] + [f"win{i}" for i in range(len(WINDOWS))]:
        p.add(act="peel", name=n)
    p.group("gain")
    p.flood("lamp", p.disc(*MOON), 2.0)
    for u, r in LAMPS:
        p.flood("lamp", p.disc(u, LAMP_V, (r + .4) * mm), 3.0)
    p.flood("lamp", p.disc(BOAT_LAMP[0], BOAT_LAMP[1], BOAT_LAMP[2] + .4 * mm), 3.0)
    for u, v in WINDOWS:
        p.flood("ember", p.rect(u - .002, v - .0035, u + .002, v + .0035), 4)
    p.flood("ember", p.rect(.699, .780, .713, .792), 3)
    if stars:
        p.group("stars")
        for k in range(22):
            u, v = rng.random(), .02 + .24 * rng.random() ** 1.5
            if (u - MOON[0]) ** 2 + (v - MOON[1]) ** 2 > .012:
                p.flood("white", p.disc(u, v, (0.22 + 0.2 * rng.random()) * mm), 6)
    return p.acts

def run(W, H, mode, out, upto=None):
    from goo.journal.session import Session
    register()
    dx = WM / W
    s = Session(dict(size=[W, H], dx=f"{dx / mm:.6g} mm", ground={"name": "paper", "porosity": 0}, substances=SUBS,
                     air={"humidity": 0.5}), keep=0, atomic=False)
    water = 0.0 if mode == "dry" else 0.6
    for name in PARTS:
        s.do(dict(act="paint", name=name, recipe=recipe(name, water)))
        s.do(dict(act="fill", vessel=name, paint=name, volume="2 L"))
    if mode == "dry":
        s.do(dict(act="hold", region="all"))
    acts = compose(min_r=0.75 * dx if W < 600 else 0.0, stars=W >= 600)
    t0 = time.time(); g = None
    for a in acts:
        if "group" in a:
            if g is not None and mode == "dry":
                s.do(dict(act="settle", force=True))
            g = a["group"]; print(f"[{time.time() - t0:6.1f}s] {g}", flush=True)
            if upto and g == upto: break
            continue
        s.do(a)
    if mode == "dry":
        s.do(dict(act="settle", force=True))
    return s

def tiled_picture(s, exposure, nx=4, ny=4):
    """The whole panel, looked at tile by tile so that a look's memory is a tile's, not the panel's."""
    W, H, _ = s.size
    img = np.zeros((H, W, 3), np.uint8); flags = {}; res = 0.0
    for j in range(ny):
        for i in range(nx):
            x0, x1, y0, y1 = i * W // nx, (i + 1) * W // nx, j * H // ny, (j + 1) * H // ny
            pic = s.picture(exposure=exposure, region={"rect": [x0, y0, x1, y1]}, scale=1, cache={})
            img[y0:y1, x0:x1] = np.asarray(pic.image)[:y1 - y0, :x1 - x0]
            for f in pic.receipt.flags: flags[f.kind] = flags.get(f.kind, 0) + 1
            res = max(res, float(pic.receipt.residual.get("light", 0.0)))
    return img, flags, res

if __name__ == "__main__":
    import pickle
    from PIL import Image
    W, H, mode, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    exposure = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
    t0 = time.time()
    s = run(W, H, mode, out)
    json.dump(s.entries, open(out.replace(".png", ".journal.json"), "w"))
    pickle.dump(s, open(out.replace(".png", ".session.pkl"), "wb"), protocol=4)
    print(f"[{time.time() - t0:6.1f}s] laid and saved; looking", flush=True)
    n = 4 if W >= 600 else 1
    img, flags, res = tiled_picture(s, exposure, n, n)
    Image.fromarray(img).resize((W * max(1, 768 // H), H * max(1, 768 // H)), Image.NEAREST).save(out)
    print("saved", out, "entries", len(s.entries), "world t", s.time, "wall", round(time.time() - t0), "s; look flags", flags,
          "light residual", res)
