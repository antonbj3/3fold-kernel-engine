#!/usr/bin/env python3
"""CAD -> 2D FEM MESH (gmsh) - real CAD geometry made simulation-ready, with a QUALITY gate.

The point is to take REAL CAD geometry and make it physics-ready. AUDIT FIX:
an earlier BRepMesh version produced a DEGENERATE holed-plate mesh (158/162 sliver triangles AR~118, ZERO interior
nodes - BRepMesh only refines the curved hole edge and leaves the flat face without interior discretisation).
A downstream MMS on such a mesh is near-vacuous (P2 reproduces the quadratic regardless of element shape). It now runs
through the gmsh 2D mesher (OCC geometry + Delaunay with interior nodes) -> a genuine unstructured mesh with a quality gate.

Generates two planar CAD geometries -> 2D triangulation + mesh QUALITY gate (aspect ratio):
  - rectangle (LX x LY)            - holed plate (rectangle minus circle)

  python3 cad_to_femmesh.py
"""
import sys
from pathlib import Path

import numpy as np

LX, LY = 2.0, 1.0
HOLE_R = 0.2
MESH_SIZE = 0.08
OUT = Path("data/cad_meshes")


def _gmsh_2d(name, build_fn, expect_area):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(name)
    build_fn(gmsh)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", MESH_SIZE)
    gmsh.model.mesh.generate(2)
    ntags, ncoords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(ncoords).reshape(-1, 3)[:, :2]            # plan → (x,y)
    tag2idx = {int(t): i for i, t in enumerate(ntags)}
    etypes, _, enodes = gmsh.model.mesh.getElements(dim=2)
    tris = None
    for et, en in zip(etypes, enodes):
        if et == 2:                                            # 3-nods triangel
            tt = np.array(en, dtype=np.int64).reshape(-1, 3)
            tris = np.vectorize(lambda t: tag2idx[int(t)])(tt)
    gmsh.finalize()

    a = coords[tris[:, 0]]; b = coords[tris[:, 1]]; c = coords[tris[:, 2]]
    area = float(0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                              - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])).sum())
    e = np.stack([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                  np.linalg.norm(a - c, axis=1)], 1)
    ar = e.max(1) / np.maximum(e.min(1), 1e-12)
    rel = abs(area - expect_area) / expect_area
    interior = np.sum((coords[:, 0] > 0.05) & (coords[:, 0] < LX - 0.05)
                      & (coords[:, 1] > 0.05) & (coords[:, 1] < LY - 0.05))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", positions=coords, tris=tris, LX=LX, LY=LY, hole_r=HOLE_R)
    ok = rel < 5e-2 and float(np.median(ar)) < 5.0 and interior > 20      # QUALITY gate
    print(f"  {name}: {len(coords)} noder ({interior} inre), {len(tris)} tri, area {area:.4f} "
          f"(rel {rel:.2e}), AR median {np.median(ar):.1f}/max {ar.max():.0f} → {OUT/(name+'.npz')}")
    return ok


def _rect(gmsh):
    gmsh.model.occ.addRectangle(0, 0, 0, LX, LY)


def _holeplate(gmsh):
    r = gmsh.model.occ.addRectangle(0, 0, 0, LX, LY)
    d = gmsh.model.occ.addDisk(LX / 2, LY / 2, 0, HOLE_R, HOLE_R)
    gmsh.model.occ.cut([(2, r)], [(2, d)])


def main():
    print(f"CAD -> 2D FEM mesh (gmsh OCC, mesh size {MESH_SIZE}) - rectangle {LX}x{LY}, hole r={HOLE_R}")
    ok1 = _gmsh_2d("cad_rect", _rect, LX * LY)
    ok2 = _gmsh_2d("cad_holeplate", _holeplate, LX * LY - np.pi * HOLE_R ** 2)
    ok = ok1 and ok2
    print(f"\nVERDICT: CAD -> 2D mesh {'GENERATED + area and QUALITY verified' if ok else 'NOT OK'} - gmsh Delaunay "
          + ("with genuine interior discretisation (AR median <5, >20 interior nodes) -> a representative unstructured mesh, "
             "not sliver-degenerate. Consumed by warpfem_cad_elasticity.py + warpfem_mms_cad.py. " if ok else
             "area mismatch OR a degenerate mesh (slivers / no interior nodes). ")
          + "AUDIT FIX: replaced BRepMesh (degenerate flat faces) with gmsh + a mesh-quality gate.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
