"""The shared types every module imports: units, evidence, ledger, film. Owner: the lead."""
import numpy as np, pytest
from goo.units import um, nm, k_B, N_A, R_gas
from goo.evidence import weakest, Provenance, Derived, Bounded, derive, EVIDENCE
from goo.ledger import Ledger, Receipt, Flag
from goo.film import Bands, Light, Film, Ground, Stack, Environment


class Sub:
    def __init__(self, name, rho): self.name, self.rho = name, rho


def test_evidence_is_ordered_and_weakest_wins():
    assert weakest("proved", "computed", "exact") == "computed"
    assert weakest(*EVIDENCE) == "guessed"
    with pytest.raises(ValueError): weakest("measured")
    assert R_gas == k_B * N_A


def test_provenance_needs_a_source_and_licence_when_computed():
    Provenance("guessed")
    Provenance("computed", "refractiveindex.info: Devore 1951", "CC0")
    with pytest.raises(ValueError): Provenance("computed", "somewhere")
    with pytest.raises(ValueError): Provenance("checked")


def test_derive_takes_the_weakest_and_is_at_best_assumed_out_of_domain():
    assert derive(1.0, "proved", "computed").evidence == "computed"
    d = derive(np.ones(3), "proved", "computed", in_domain=np.array([True, False, True]))
    assert d.evidence == "assumed" and not d.everywhere_in_domain
    assert derive(1.0, "proved", "guessed", in_domain=False).evidence == "guessed"


def test_bounded_collapses_with_a_record():
    b = Bounded(np.array([1.0, 2.0]), np.array([3.0, 2.5]), "computed", "effective index")
    d, note = b.collapse("mid")
    assert np.allclose(d.value, [2.0, 2.25]) and d.evidence == "assumed" and "effective index" in note
    with pytest.raises(ValueError): Bounded(2.0, 1.0, "assumed")


def test_ledger_is_double_entry_and_measures_closure():
    L = Ledger("matter", (2,), ("water", "ochre"))
    L.post("origin", "wet", [3.0, 1.0])
    L.post("wet", "reservoir[atmosphere]", [1.25, 0.0], "evaporation")
    L.post("wet", "stratum[0]", [0.5, 1.0])
    assert L.imbalance() == 0.0
    assert np.array_equal(L.balance("wet"), [1.25, 0.0])
    assert L.residual("wet", [1.25, 0.0]) == 0.0
    assert L.residual("wet", [1.25, 1e-9]) > 0.0
    assert np.array_equal(L.held("stratum"), [0.5, 1.0])
    with pytest.raises(ValueError): L.post("wet", "wet", [1.0, 0.0])
    with pytest.raises(ValueError): L.post("wet", "ground", [np.nan, 0.0])
    f = L.fork(); f.post("origin", "wet", [1.0, 1.0])
    assert np.array_equal(L.balance("wet"), [1.25, 0.0]) and f.digest() != L.digest()


def test_receipts_add_and_digest():
    L = Ledger("energy")
    m = L.mark()
    L.post("work[brush]", "stored", 5.0); L.post("stored", "dissipated[viscous]", 3.0)
    L.post("stored", "dissipated[numerical]", 0.25)
    r1 = Receipt("step", 0.0, 1.0, L.since(m), {"energy": 1e-15}, {"gravity": (2.0, 3.0)},
                 report={"D_num time": 0.25, "max slope": 0.02})
    m = L.mark(); L.post("stored", "dissipated[viscous]", 1.0)
    r2 = Receipt("step", 1.0, 2.0, L.since(m), {"energy": 3e-15}, {"gravity": (3.0, 2.5)},
                 (Flag("privilege", "painter", "hold", 0.0),), {"D_num time": 0.5, "max slope": 0.01})
    r = r1 + r2
    assert (r.t0, r.t1) == (0.0, 2.0) and r.residual["energy"] == 3e-15 and r.F["gravity"] == (2.0, 2.5)
    assert r.moved("energy", dst="dissipated") == 4.25 and r.moved("energy", src="work") == 5.0
    assert r.moved("matter") == 0.0 and len(r.flagged("privilege")) == 1
    assert r.report == {"D_num time": 0.75, "max slope": 0.02}
    assert r.digest() == (r1 + r2).digest() != r1.digest()
    with pytest.raises(ValueError): Flag("pearlescent", "x")


def test_film_and_stack_are_plain_data():
    water, ochre = Sub("water", 998.0), Sub("ochre", 3500.0)
    f = Film.uniform({water: 40 * um, ochre: 10 * um}, H=2, W=3)
    assert f.shape == (2, 3) and np.allclose(f.h, 50 * um) and np.allclose(f.phi[1], 0.2)
    assert np.allclose(f.mass(dx=1e-3), [998.0 * 40e-6 * 6e-6, 3500.0 * 10e-6 * 6e-6])
    assert f.index("ochre") == 1 and np.all(Film.empty((water,), 2, 3).phi == 0.0)
    with pytest.raises(ValueError): Film((water, water), np.zeros((2, 1, 1)))
    g = Ground(np.zeros((2, 3)), Sub("cellulose", 1500.0), 0.5, 5 * um, 200 * um, (water,))
    assert g.capacity == 100 * um and g.held.shape == (1, 2, 3)
    s = Stack(g, [f, f.copy()], dx=1e-3)
    assert np.allclose(s.top, 100 * um)
    b = Bands.uniform().with_band(1550 * nm, 100 * nm)
    assert b.B == 19 and np.isclose(b.centres[0], 390 * nm)
    Light(b, np.ones(b.B))
    with pytest.raises(ValueError): Light(b, np.ones(3))


def test_environment_is_plain_data():
    e = Environment(activity={"water": 0.5})
    assert e.gravity[2] > 9.8 and e.with_(activity={"water": 0.9}).activity["water"] == 0.9 and e.activity["water"] == 0.5
    with pytest.raises(ValueError): Environment(gravity=(0, 1))
    with pytest.raises(ValueError): Environment(activity={"water": 0.0})
