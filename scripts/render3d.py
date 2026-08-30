"""Render an elevation field in 3D, with the real furniture on it.

Every figure so far has been a matplotlib plan view, which is the wrong picture
for a paper about floors that are not flat: a top-down drawing is exactly the
projection that hides the thing being claimed. This builds the scene as geometry
— tier slabs at their own heights, riser faces at every transition, walls — puts
the actual 3D-FUTURE meshes on it, and renders it.

Blender arrives as ``bpy``, which came with the Infinigen install; no separate
renderer is needed. Run it with the ``infinigen`` interpreter, not ``reroom``.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys

import numpy as np

FUTURE_ROOTS = sorted(glob.glob(
    "/home/gino/data/reroom/3D-FUTURE/3D-FUTURE-model-part*"))
SLAB = 0.12          # how thick a tier slab is drawn
WALL_T = 0.10


def asset_index() -> dict:
    """jid -> directory holding normalized_model.obj."""
    idx = {}
    for root in FUTURE_ROOTS:
        for d in os.scandir(root):
            if d.is_dir():
                idx[d.name] = d.path
    return idx


# --------------------------------------------------------------------------
# scene construction
# --------------------------------------------------------------------------

def _clear(bpy):
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _mat(bpy, name, rgb, rough=0.75, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return m


def _cap_triangles(poly):
    """Triangulate a polygon that may be concave and may have holes.

    The datum tier is the room minus the elevated region, which comes out of
    shapely as a ~90-vertex ring that wraps most of the way around a notch.
    Handing that to bmesh as a single n-gon produces a spurious triangle across
    the notch -- visible in the first render of every cross-shaped room. Delaunay
    over the vertices followed by an inside test is exact for this case and
    needs no extra dependency.
    """
    from shapely.geometry import Polygon
    from shapely.ops import triangulate as _tri

    tris = []
    for t in _tri(poly):
        if poly.contains(t.representative_point()):
            tris.append(np.asarray(t.exterior.coords)[:3])
    if not tris:                       # degenerate sliver; fall back to the ring
        return [np.asarray(poly.exterior.coords)[:-1]]
    return tris


def _rings(poly):
    yield np.asarray(poly.exterior.coords)[:-1]
    for r in poly.interiors:
        yield np.asarray(r.coords)[:-1]


def _prism(bpy, poly, z0, z1, name, mat):
    """A closed prism over a shapely polygon, from z0 to z1."""
    import bmesh
    from shapely.geometry import Polygon

    if not isinstance(poly, Polygon):
        poly = Polygon(np.asarray(poly, dtype=float))
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if poly.geom_type != "Polygon":                 # keep the largest lobe
        poly = max(poly.geoms, key=lambda g: g.area)

    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    bm = bmesh.new()

    for tri in _cap_triangles(poly):
        bm.faces.new([bm.verts.new((float(x), float(y), z1)) for x, y in tri])
        bm.faces.new([bm.verts.new((float(x), float(y), z0))
                      for x, y in tri[::-1]])
    for ring in _rings(poly):
        n = len(ring)
        lo = [bm.verts.new((float(x), float(y), z0)) for x, y in ring]
        hi = [bm.verts.new((float(x), float(y), z1)) for x, y in ring]
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new([lo[i], lo[j], hi[j], hi[i]])

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bm.normal_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    ob.data.materials.append(mat)
    return ob


def build_room(bpy, es, mats):
    """Tier slabs at their heights, plus walls around the outline."""
    from shapely.geometry import Polygon

    # Every tier is drawn from one common base up to its own height, so the
    # side of a tier *is* the riser.  Drawing each as a thin independent slab
    # leaves air between them and the drop stops being legible, which defeats
    # the purpose of rendering this at all.
    hs = [t.height for t in es.field.tiers]
    base = min(hs) - SLAB
    for t in es.field.tiers:
        if len(np.asarray(t.polygon, float)) < 3:
            continue
        _prism(bpy, t.shp, base, t.height,          # t.shp carries the holes
               f"tier_{t.tid}", mats["floor"] if abs(t.height) < 1e-6
               else mats["tier"])

    room = np.asarray(es.room.polygon, float)
    h = float(getattr(es.room, "height", 2.8))
    inner = Polygon(room)
    outer = inner.buffer(WALL_T, join_style=2)
    if outer.geom_type == "Polygon":
        base = min(t.height for t in es.field.tiers) - SLAB
        # A dollhouse cut: full-height walls hide the floor, which is the one
        # thing these figures exist to show.  They are drawn knee-high instead,
        # enough to read the room's shape without occluding the tiers.
        wh = min(h, 1.05)
        wall = _prism(bpy, outer, base, base + wh, "walls", mats["wall"])
        cut = _prism(bpy, inner, base - 0.05, base + wh + 0.05, "wall_cut",
                     mats["wall"])
        m = wall.modifiers.new("cut", "BOOLEAN")
        m.operation = "DIFFERENCE"
        m.object = cut
        bpy.context.view_layer.objects.active = wall
        bpy.ops.object.modifier_apply(modifier="cut")
        bpy.data.objects.remove(cut, do_unlink=True)


def place_objects(bpy, es, idx, max_objects=40):
    """Import the real 3D-FUTURE mesh for each object and fit it to the box."""
    from elevate3d.core.scene import TIER
    placed = 0
    for o in es.objects:
        if placed >= max_objects:
            break
        # ceiling-hung objects read as slabs floating in mid-air once the
        # ceiling is cut away for the dollhouse view
        if es.dz.get(o.oid, 0.0) > 1.0:
            continue
        jid = getattr(o, "jid", None)
        d = idx.get(jid) if jid else None
        path = os.path.join(d, "normalized_model.obj") if d else None
        sz = np.asarray(o.size, float)
        z = float(o.position[2])
        if not path or not os.path.exists(path):
            continue
        before = set(bpy.data.objects)
        try:
            bpy.ops.wm.obj_import(filepath=path, forward_axis="NEGATIVE_Z",
                                  up_axis="Y")
        except Exception:
            continue
        new = [x for x in bpy.data.objects if x not in before]
        if not new:
            continue
        bpy.ops.object.select_all(action="DESELECT")
        for x in new:
            x.select_set(True)
        bpy.context.view_layer.objects.active = new[0]
        if len(new) > 1:
            bpy.ops.object.join()
        ob = bpy.context.view_layer.objects.active

        # the normalised model is unit-ish; scale its bounding box onto ours
        bb = np.array([list(v) for v in ob.bound_box], dtype=float)
        ext = bb.max(0) - bb.min(0)
        ext[ext < 1e-6] = 1.0
        ob.scale = (sz[0] / ext[0], sz[1] / ext[1], sz[2] / ext[2])
        bpy.context.view_layer.update()
        bb2 = np.array([ob.matrix_world @ v.co for v in ob.data.vertices[:2000]],
                       dtype=float) if len(ob.data.vertices) else None
        ob.rotation_euler = (0.0, 0.0, float(o.yaw))
        bpy.context.view_layer.update()
        # sit it on its support: recompute the world box after the rotation
        vs = np.array([ob.matrix_world @ v.co for v in ob.data.vertices],
                      dtype=float)
        centre = (vs.min(0) + vs.max(0)) / 2.0
        ob.location = (float(o.xy[0]) - centre[0] + ob.location[0],
                       float(o.xy[1]) - centre[1] + ob.location[1],
                       z - vs.min(0)[2] + ob.location[2])
        ob.name = f"obj_{o.oid}"
        placed += 1
    return placed


def setup_camera(bpy, es, azim=-52.0, elev=34.0, margin=1.06):
    room = np.asarray(es.room.polygon, float)
    c = room.mean(0)
    rad = float(np.abs(room - c).max())
    hs = [t.height for t in es.field.tiers]
    zc = (min(hs) + max(hs)) / 2.0 + 0.55

    a, e = math.radians(azim), math.radians(elev)
    # frame from the lens rather than a guessed multiplier, or the room ends up
    # a stamp in the middle of an empty plate
    sensor, lens = 36.0, 34.0
    hfov = 2.0 * math.atan(sensor / (2.0 * lens))
    dist = (rad * margin) / math.tan(hfov / 2.0)
    setup_camera._dist = dist
    loc = (c[0] + dist * math.cos(e) * math.cos(a),
           c[1] + dist * math.cos(e) * math.sin(a),
           zc + dist * math.sin(e))
    cam_d = bpy.data.cameras.new("cam")
    cam_d.lens = 34.0
    cam = bpy.data.objects.new("cam", cam_d)
    bpy.context.collection.objects.link(cam)
    cam.location = loc
    # hand-rolling the Euler angles for a camera pointed at a target is the
    # classic way to render a grey rectangle; let mathutils do it
    from mathutils import Vector
    d = Vector((c[0] - loc[0], c[1] - loc[1], zc - loc[2]))
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam, np.array([c[0], c[1], zc]), (a, e)


def fit_camera(bpy, cam, target, angles, pad=1.02, iters=14):
    """Pull the camera back until everything is inside the frame.

    Computing a distance from the lens gets the scale roughly right and still
    crops an L-shaped room, because the bounding radius about the centroid is
    not the radius about the view axis.  Projecting the actual geometry and
    expanding is simpler than being clever.
    """
    from bpy_extras.object_utils import world_to_camera_view
    sc = bpy.context.scene
    pts = []
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        for v in ob.bound_box:
            pts.append(ob.matrix_world @ __import__("mathutils").Vector(v))
    if not pts:
        return
    a, e = angles
    dist = float(np.linalg.norm(np.array(cam.location) - target))
    for _ in range(iters):
        bpy.context.view_layer.update()
        uv = [world_to_camera_view(sc, cam, p) for p in pts]
        u = max(max(abs(p.x - 0.5) for p in uv), 1e-6) * 2.0
        v = max(max(abs(p.y - 0.5) for p in uv), 1e-6) * 2.0
        need = max(u, v) * pad
        if 0.92 <= need <= 1.0:
            break
        dist *= need
        cam.location = (target[0] + dist * math.cos(e) * math.cos(a),
                        target[1] + dist * math.cos(e) * math.sin(a),
                        target[2] + dist * math.sin(e))
        from mathutils import Vector
        dvec = Vector((target[0] - cam.location[0], target[1] - cam.location[1],
                       target[2] - cam.location[2]))
        cam.rotation_euler = dvec.to_track_quat("-Z", "Y").to_euler()


def setup_light(bpy, es):
    room = np.asarray(es.room.polygon, float)
    c = room.mean(0)
    rad = float(np.abs(room - c).max())
    sun_d = bpy.data.lights.new("sun", type="SUN")
    sun_d.energy = 2.6
    sun_d.angle = math.radians(12)
    sun = bpy.data.objects.new("sun", sun_d)
    bpy.context.collection.objects.link(sun)
    sun.location = (c[0] - rad, c[1] - rad, 6.0)
    sun.rotation_euler = (math.radians(48), 0.0, math.radians(40))

    area_d = bpy.data.lights.new("fill", type="AREA")
    area_d.energy = 110.0
    area_d.size = max(rad * 2.0, 3.0)
    area = bpy.data.objects.new("fill", area_d)
    bpy.context.collection.objects.link(area)
    area.location = (c[0], c[1], 3.4)

    # A second sun from the opposite side, aimed low. Raising the world strength
    # instead lifts the furniture and the walls together, and the walls are
    # already near-white -- they wash straight into the background. A directional
    # fill lifts the shaded faces of the furniture and leaves the walls shaded.
    back_d = bpy.data.lights.new("back", type="SUN")
    back_d.energy = 1.1
    back_d.angle = math.radians(35)
    back = bpy.data.objects.new("back", back_d)
    bpy.context.collection.objects.link(back)
    back.location = (c[0] + rad, c[1] + rad, 5.0)
    back.rotation_euler = (math.radians(62), 0.0, math.radians(-140))

    w = bpy.context.scene.world or bpy.data.worlds.new("w")
    bpy.context.scene.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.97, 0.97, 0.98, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.50


def render(bpy, out, res=(900, 620), samples=48):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE_NEXT"
    try:
        sc.eevee.taa_render_samples = samples
    except Exception:
        pass
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = True
    # Blender 4.2 defaults to the AgX view transform, which renders a white
    # world as mid grey.  These are diagrams, not photographs.
    try:
        sc.view_settings.view_transform = "Standard"
    except Exception:
        pass
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = out
    bpy.ops.render.render(write_still=True)


def render_scene(es, out, idx, res=(900, 620), azim=-52.0, elev=34.0):
    import bpy
    _clear(bpy)
    mats = {
        # White geometry on a white world renders as nothing at all.  These
        # carry actual tone so the floor reads against the background and the
        # datum reads against the elevated tier.
        "floor": _mat(bpy, "floor", (0.84, 0.78, 0.67), 0.9),
        "tier": _mat(bpy, "tier", (0.38, 0.53, 0.63), 0.75),
        "wall": _mat(bpy, "wall", (0.62, 0.61, 0.59), 0.95),
    }
    build_room(bpy, es, mats)
    n = place_objects(bpy, es, idx)
    cam, target, angles = setup_camera(bpy, es, azim, elev)
    fit_camera(bpy, cam, target, angles)
    setup_light(bpy, es)
    render(bpy, out, res)
    return n
