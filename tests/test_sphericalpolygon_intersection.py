# STDLIB
import codecs

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from sphersgeo import MultiSphericalPolygon, SphericalPolygon, SphericalPoint

from .helpers import ROOT_DIR, resolve_imagename


@pytest.mark.parametrize(
    "center_a,radius_a,",
    [
        ((-173, 78), 13),
        ((-170, 43), 11),
        ((-177, 84), 14),
    ],
)
@pytest.mark.parametrize(
    "center_b,radius_b",
    [
        ((178, -31), 9),
        ((173, -44), 11),
        ((175, -85), 5),
    ],
)
def test_nonintersection(center_a, radius_a, center_b, radius_b):
    a = SphericalPolygon.from_cone(
        center_a,
        radius_a,
        steps=16,
    )

    b = SphericalPolygon.from_cone(
        center_b,
        radius_b,
        steps=16,
    )

    assert a.intersection(b) is None


def test_intersects():
    a = SphericalPolygon([(20.0, 5.0), (25.0, 5.0), (25.0, 10.0), (20.0, 10.0)])
    b = SphericalPolygon([(18.0, 6.0), (21.0, 6.0), (21.0, 7.0), (18.0, 7.0)])

    assert a.intersects(b)
    assert a.intersection(b).area > 0.0


def test_intersects_simple():
    lon1 = np.array([-10, 10, 10, -10, -10], dtype=float)
    lat1 = np.array([30, 30, 0, 0, 30], dtype=float)
    p1 = SphericalPolygon(np.stack((lon1, lat1), axis=1))

    lon2 = np.array([-5, 15, 15, -5, -5], dtype=float)
    lat2 = np.array([20, 20, -10, -10, 20], dtype=float)
    p2 = SphericalPolygon(np.stack((lon2, lat2), axis=1))

    assert p1.intersects(p2)

    # Make sure it isn't order-dependent
    p1r = SphericalPolygon(np.stack((lon1[::-1], lat1[::-1]), axis=1))
    p2r = SphericalPolygon(np.stack((lon2[::-1], lat2[::-1]), axis=1))

    assert p1r.intersects(p2r)


def test_intersects_poly_fully_contained():
    lon1 = np.array([-10, 10, 10, -10, -10], dtype=float)
    lat1 = np.array([30, 30, 0, 0, 30], dtype=float)
    p1 = SphericalPolygon(np.stack((lon1, lat1), axis=1))

    lon2 = np.array([-5, 5, 5, -5, -5], dtype=float)
    lat2 = np.array([20, 20, 10, 10, 20], dtype=float)
    p2 = SphericalPolygon(np.stack((lon2, lat2), axis=1))

    assert p1.intersects(p2)

    # Make sure it isn't order-dependent
    p1r = SphericalPolygon(np.stack((lon1[::-1], lat1[::-1]), axis=1))

    p2r = SphericalPolygon(np.stack((lon2[::-1], lat2[::-1]), axis=1))

    assert p1r.intersects(p2r)


def test_hard_intersects_poly():
    lon1 = np.array([-10, 10, 10, -10, -10], dtype=float)
    lat1 = np.array([30, 30, 0, 0, 30], dtype=float)
    p1 = SphericalPolygon(np.stack((lon1, lat1), axis=1))

    lon2 = np.array([-20, 20, 20, -20, -20], dtype=float)
    lat2 = np.array([20, 20, 10, 10, 20], dtype=float)
    p2 = SphericalPolygon(np.stack((lon2, lat2), axis=1))

    assert p1.intersects(p2)

    # Make sure it isn't order-dependent
    p1r = SphericalPolygon(np.stack((lon1[::-1], lat1[::-1]), axis=1))
    p2r = SphericalPolygon(np.stack((lon2[::-1], lat2[::-1]), axis=1))

    assert p1r.intersects(p2r)


def test_not_intersects_poly():
    lon1 = np.array([-10, 10, 10, -10, -10], dtype=float)
    lat1 = np.array([30, 30, 5, 5, 30], dtype=float)
    p1 = SphericalPolygon(np.stack((lon1, lat1), axis=1))

    lon2 = np.array([-20, 20, 20, -20, -20], dtype=float)
    lat2 = np.array([-20, -20, -10, -10, -20], dtype=float)
    p2 = SphericalPolygon(np.stack((lon2, lat2), axis=1))

    assert not p1.intersects(p2)

    # Make sure it isn't order-dependent
    lon1 = lon1[::-1]
    lat1 = lat1[::-1]
    p1 = SphericalPolygon(np.stack((lon1, lat1), axis=1))

    lon2 = lon2[::-1]
    lat2 = lat2[::-1]
    p2 = SphericalPolygon(np.stack((lon2, lat2), axis=1))

    assert not p1.intersects(p2)


def test_difficult_intersections():
    # Tests a number of intersections of real data that have been
    # problematic in previous revisions of spherical_geometry

    # def test_intersection(polys):
    #     A, B = polys
    #     A.intersection(B)

    fname = resolve_imagename(ROOT_DIR, "difficult_intersections.txt")
    with open(fname, "rb") as fd:
        lines = fd.readlines()

    def to_array(line):
        x = np.frombuffer(codecs.decode(line.strip(), "hex_codec"), dtype="<f8")
        return x.reshape((len(x) // 3, 3))

    for i in range(0, len(lines), 4):
        p1_points, p1_inside, p2_points, p2_inside = [
            to_array(line) for line in lines[i : i + 4]
        ]
        p1 = SphericalPolygon(p1_points, p1_inside)
        p2 = SphericalPolygon(p2_points, p2_inside)
        # yield test_intersection, (p1, p2)
        p1.intersection(p2)


def test_self_intersection():
    # Tests intersection between a disjoint polygon and itself
    p1 = SphericalPolygon(
        [
            (150.15056635, 2.33675579),
            (150.18472797, 2.33675454),
            (150.18472641, 2.30262137),
            (150.15056557, 2.3026226),
            (150.15056635, 2.33675579),
        ]
    )
    p2 = SphericalPolygon(
        [
            (150.18472955, 2.37105428),
            (150.18472798, 2.33692121),
            (150.15056635, 2.33692245),
            (150.15056714, 2.37105554),
            (150.18472955, 2.37105428),
        ]
    )

    # create a union polygon
    p12 = p2.union(p1)
    # asserts self-intersection is same as original
    p12int = p12.intersection(p12)
    assert p12int.area == p12.area


def test_ordering():
    nrepeat = 10

    p1 = SphericalPolygon(
        [
            (
                3.532808036921135653e-01,
                6.351523005458726834e-01,
                -6.868582305351954576e-01,
            ),
            (
                3.532781068942476010e-01,
                6.351564219435104075e-01,
                -6.868558064493115456e-01,
            ),
            (
                3.529538811375814156e-01,
                6.351027504797477352e-01,
                -6.870720880104047579e-01,
            ),
            (
                3.533428330964511477e-01,
                6.345142927049303161e-01,
                -6.874157800432978416e-01,
            ),
            (
                3.533486351814376647e-01,
                6.345151843837375516e-01,
                -6.874119745843003670e-01,
            ),
            (
                3.533513056857608414e-01,
                6.345111416839894769e-01,
                -6.874143334620310686e-01,
            ),
            (
                3.536740696809928530e-01,
                6.345607036635456666e-01,
                -6.872025653337667794e-01,
            ),
            (
                3.536713200704008631e-01,
                6.345649108795897719e-01,
                -6.872000954889618818e-01,
            ),
            (
                3.536761865498951884e-01,
                6.345656515431040701e-01,
                -6.871969069700470945e-01,
            ),
            (
                3.536788213460497765e-01,
                6.345616140129455296e-01,
                -6.871992792142280759e-01,
            ),
            (
                3.540056257094351122e-01,
                6.346113105009757449e-01,
                -6.869850810245486938e-01,
            ),
            (
                3.536200722272911379e-01,
                6.352081961257413090e-01,
                -6.866319189293832448e-01,
            ),
            (
                3.536142814048366390e-01,
                6.352072452054380314e-01,
                -6.866357809093986964e-01,
            ),
            (
                3.536116196666648781e-01,
                6.352113634102898310e-01,
                -6.866333419163089813e-01,
            ),
            (
                3.532833767830895755e-01,
                6.351574192193063517e-01,
                -6.868521736876195272e-01,
            ),
            (
                3.532861440234288386e-01,
                6.351531838825796861e-01,
                -6.868546669018701367e-01,
            ),
            (
                3.532808036921135653e-01,
                6.351523005458726834e-01,
                -6.868582305351954576e-01,
            ),
        ],
        (3.536414047913637448e-01, 6.348851549491377755e-01, -6.869196436573932196e-01),
    )

    p2 = SphericalPolygon(
        [
            (
                3.529249199274748783e-01,
                6.356925960489819838e-01,
                -6.865412764158403958e-01,
            ),
            (
                3.533126219535084322e-01,
                6.351003877952851040e-01,
                -6.868898664200949744e-01,
            ),
            (
                3.533173735956686712e-01,
                6.351012981906917210e-01,
                -6.868865805589428053e-01,
            ),
            (
                3.529301898742857047e-01,
                6.356935934402119237e-01,
                -6.865376437853726310e-01,
            ),
            (
                3.532584388080926563e-01,
                6.357475490961038700e-01,
                -6.863188247667159070e-01,
            ),
            (
                3.536441982306618437e-01,
                6.351510082118909661e-01,
                -6.866723948326530769e-01,
            ),
            (
                3.533173735956686712e-01,
                6.351012981906917210e-01,
                -6.868865805589428053e-01,
            ),
            (
                3.533126219535084322e-01,
                6.351003877952851040e-01,
                -6.868898664200949744e-01,
            ),
            (
                3.529898380712340189e-01,
                6.350508125724935171e-01,
                -6.871016225198859351e-01,
            ),
            (
                3.526006883384300017e-01,
                6.356389133339014341e-01,
                -6.867575456003104373e-01,
            ),
            (
                3.529249199274748783e-01,
                6.356925960489819838e-01,
                -6.865412764158403958e-01,
            ),
        ],
        (3.532883212044564125e-01, 6.354215160430938258e-01, -6.866053153377369433e-01),
    )

    areas = []
    for i in range(nrepeat):
        p12 = p1.intersection(p2)
        areas.append(p12.area)
    areas = np.array(areas)
    assert_array_almost_equal(areas[:-1], areas[1:])

    def roll_polygon(P, i):
        polygons = []
        for p in P.polygons:
            points = p.points
            points = np.roll(points[:-1], i, 0)
            points = np.append(points, [points[0]], 0)
            p = SphericalPolygon(points, p.inside)
            polygons.append(p)
        return MultiSphericalPolygon(polygons)

    p1_areas = []
    p2_areas = []
    p12_areas = []
    for i in range(nrepeat):
        p1s = roll_polygon(p1, i)
        p2s = roll_polygon(p2, i)

        p12 = p1s.intersection(p2s)

        p1_areas.append(p1.area)
        p2_areas.append(p2.area)
        p12_areas.append(p12.area)

        for j in range(nrepeat):
            p12s = roll_polygon(p12, j)
            p12_areas.append(p12s.area)

    p1_areas = np.array(p1_areas)
    p2_areas = np.array(p2_areas)
    p12_areas = np.array(p12_areas)
    assert_array_almost_equal(p1_areas[:-1], p1_areas[1:])
    assert_array_almost_equal(p2_areas[:-1], p2_areas[1:])
    assert_array_almost_equal(p12_areas[:-1], p12_areas[1:])


def test_intersection_crash():
    # Reported by Darren White

    p1 = SphericalPolygon(
        [
            (0.3583051, 0.04329773, 0.9326),
            (0.34935525, 0.09059884, 0.9326),
            (0.41458613, 0.10294118, 0.90416893),
            (0.42353598, 0.05564007, 0.90416893),
            (0.3583051, 0.04329773, 0.9326),
        ],
        (0.38681003, 0.07318841, 0.91925049),
    )

    p2 = SphericalPolygon(
        [
            (0.4246086, 0.04379716, 0.90431706),
            (0.41771239, 0.08700592, 0.90440386),
            (0.35406757, 0.06995283, 0.9326),
            (0.34465183, 0.10538573, 0.93279631),
            (0.29364251, 0.08598585, 0.95204018),
            (0.27687092, 0.13007964, 0.95206186),
            (0.23924242, 0.10865033, 0.96486174),
            (0.21881505, 0.14456566, 0.96499779),
            (0.19972446, 0.12927807, 0.97128643),
            (0.17211517, 0.16374755, 0.97137177),
            (0.16418552, 0.15451288, 0.97425299),
            (0.12763925, 0.18584761, 0.97425299),
            (0.17303981, 0.2387991, 0.95552719),
            (0.20958607, 0.20746437, 0.95552719),
            (0.1861715, 0.18013143, 0.96586378),
            (0.22383787, 0.21031346, 0.95166426),
            (0.253951, 0.17275425, 0.95166426),
            (0.25269325, 0.17174401, 0.95218177),
            (0.27532947, 0.18462841, 0.94345431),
            (0.29916705, 0.1428042, 0.94345431),
            (0.28233717, 0.13319738, 0.95002325),
            (0.34001511, 0.15514004, 0.92753506),
            (0.35358851, 0.11967283, 0.92771417),
            (0.40573405, 0.1336354, 0.90416893),
            (0.41710657, 0.0913497, 0.90425513),
            (0.48033933, 0.10150229, 0.87118965),
            (0.48707009, 0.05997741, 0.871301),
            (0.4841862, 0.08268655, 0.87104917),
            (0.54534645, 0.09006864, 0.8333576),
            (0.55111516, 0.04227516, 0.8333576),
            (0.48995491, 0.03489307, 0.87104917),
            (0.48779298, 0.05394194, 0.87129115),
            (0.4246086, 0.04379716, 0.90431706),
        ],
        (0.16877166, 0.19684143, 0.96579997),
    )

    assert p1.intersects(p2)
    assert p1.overlaps(p2)


@pytest.mark.xfail(reason="currently there is no solution to get this to pass")
def test_intersection_crash_similar_poly():
    p1 = SphericalPolygon(
        (
            [
                (-0.1094946215827374, -0.8592766830993238, -0.499654390280199),
                (-0.1089683641318892, -0.8595220381654031, -0.4993473355555343),
                (-0.108610535224965, -0.8593183788298407, -0.4997756250993051),
                (-0.1091500557209236, -0.8590667764452905, -0.5000905307482003),
                (-0.1094946215827374, -0.8592766830993238, -0.499654390280199),
            ],
            (-0.1090595793730483, -0.8592979843505629, -0.4997128998115153),
        )
    )

    p2 = SphericalPolygon(
        (
            [
                (-0.1094946213367254, -0.8592766831114167, -0.4996543903133135),
                (-0.1089683641834766, -0.859522038038747, -0.4993473357622887),
                (-0.1086105354789061, -0.8593183788183577, -0.4997756250638628),
                (-0.109150055669766, -0.8590667765760884, -0.5000905305346783),
                (-0.1094946213367254, -0.8592766831114167, -0.4996543903133135),
            ],
            (-0.1090595793730483, -0.8592979843505629, -0.4997128998115153),
        )
    )

    assert p1.intersects(p2)
    assert p1.overlaps(p2)

    p12 = p1.intersection(p2)

    assert p12 is not None

    assert_allclose(
        np.sort(p1.vertices.xyzs[:-1], axis=0),
        np.sort(p12.vertices.xyzs[:-1], axis=0),
        rtol=0,
        atol=1e-15,
    )


def test_complement_regression():
    """https://github.com/spacetelescope/spherical_geometry/issues/278"""

    p1 = SphericalPolygon.from_cone((90, 0), 100)
    p2 = SphericalPolygon.from_cone((270, 0), 100)

    origin = SphericalPoint((0, 0))
    assert p1.contains(origin)
    assert p2.contains(origin)

    p12 = p1.intersection(p2)

    assert p12.contains(origin)
