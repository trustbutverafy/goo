"""Quick physics checks on flat targets before touching the painting."""
import numpy as np, time
from goo.sensors.coherent import Surface, Geometry
from goo.sensors.lidar import CELL
H, W = 80, 80                                   # 20 x 20 mm flat patch
rng = np.random.default_rng(0)
def flat(refl=0.9, rough=1.5, tilt=0.0):
    Z = np.zeros((H, W)) + tilt * (np.arange(W) * CELL)[None, :]
    return Surface(Z, np.full((H, W), refl), np.full((H, W), rough))
def run(surf, theta_deg, **kw):
    th = np.radians(theta_deg)
    pos = (10 - 150 * np.sin(th), 10, 150 * np.cos(th))
    g = Geometry(surf, pos, spacing=0.1, aim_box=(6, 14, 6, 14), seed=1)
    return g
if __name__ == "__main__":
  t = time.time()
  print("thermal calibration: one scatterer-equivalent? no - speckle on. Normal incidence, white, rough 1.5 um")
  for th in [0, 20, 40, 60, 75]:
      g = run(flat(), th)
      row = [f"{th:2d} deg  spread {np.nanmedian(g.spread)*1000:6.1f} um"]
      for lab, kw in [("incoh", dict(mode="incoherent")), ("100G", dict()), ("100G x4", dict(lams=(1530e-9, 1545e-9, 1560e-9, 1575e-9))),
                      ("1T", dict(bandwidth=1e12)), ("1T x4", dict(bandwidth=1e12, lams=(1530e-9, 1545e-9, 1560e-9, 1575e-9)))]:
          e = g.errors(g.measure(**kw)["R"])["e_beam"][g.valid] * 1000
          row.append(f"{lab}: rms {np.sqrt(np.mean(e**2)):6.1f} p99 {np.percentile(np.abs(e),99):6.0f} max {np.abs(e).max():5.0f}")
      print(" | ".join(row), flush=True)
  print(f"{time.time()-t:.0f}s")
