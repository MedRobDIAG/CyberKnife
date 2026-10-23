# ============================================================================
#  tumor_3d.py - Da maschera voxel -> volume + modello 3D (mesh STL) del tumore.
#
#  Serve come ponte verso la Fase 2 (path-planning / "riempimento" con i raggi):
#    1. VOLUME    : quanti mm^3 / cm^3 occupa il tumore (per la dose/planning).
#    2. MESH 3D   : superficie del tumore ricostruita con Marching Cubes ed
#                   esportata in STL nello spazio FISICO reale (mm). Lo STL e'
#                   lo standard per simulatori robotici / CAD / Blender.
#    3. PREVIEW   : rendering 3D statico (.png) per un colpo d'occhio.
#
#  USO:
#    python tumor_3d.py --mask "outputs\lung_001_seg.nii.gz"
# ============================================================================
import os
import argparse

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from scipy import ndimage
from skimage import measure
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import config


def largest_component(mask, affine=None, min_lesion_mm3=None):
    """Scarta le componenti troppo piccole per essere lesioni.

    Sostituisce la vecchia regola "tieni la componente piu' grande". Quella
    regola non poteva mai restituire una maschera vuota: acceso un solo voxel,
    quel voxel diventava la lesione. E se un falso positivo superava in volume
    la lesione vera, la mesh e il bersaglio ci saltavano sopra in silenzio.

    Il criterio ora e' un volume FISICO (mm3), interpretabile e verificabile:
    100 mm3 = sfera di 5.8 mm, la piu' piccola lesione realmente annotata nel
    dataset. Serve l'affine per convertire i voxel in mm3; senza, si ricade sul
    comportamento storico per non cambiare il risultato di chi la chiama senza.
    """
    labeled, num = ndimage.label(mask)
    if num <= 1:
        return np.asarray(mask).astype(np.uint8)

    ids = np.arange(1, num + 1)
    sizes = np.asarray(ndimage.sum(np.ones_like(mask, dtype=np.uint8), labeled,
                                   index=ids))
    if affine is None:
        keep_ids = ids[[int(np.argmax(sizes))]]
        print(f"[3d] {num} componenti, nessuna affine -> tenuta la maggiore.")
    else:
        min_mm3 = (config.POSTPROC.min_lesion_mm3 if min_lesion_mm3 is None
                   else min_lesion_mm3)
        vox_mm3 = abs(np.linalg.det(np.asarray(affine)[:3, :3]))
        keep_ids = ids[sizes * vox_mm3 >= min_mm3]
        print(f"[3d] {num} componenti -> {len(keep_ids)} sopra {min_mm3:.0f} mm3.")
    return np.isin(labeled, keep_ids).astype(np.uint8)


def compute_volume(mask, affine):
    """Volume del tumore in mm^3 e cm^3.

    Il volume di UN voxel = |det(parte lineare 3x3 dell'affine)|: vale per
    qualsiasi orientamento/spacing anisotropo. Moltiplicato per il numero di
    voxel del tumore da' il volume totale.
    """
    voxel_vol_mm3 = abs(np.linalg.det(affine[:3, :3]))
    n_voxels = int(mask.sum())
    vol_mm3 = n_voxels * voxel_vol_mm3
    vol_cm3 = vol_mm3 / 1000.0   # 1 cm^3 = 1000 mm^3 = 1 mL
    return n_voxels, voxel_vol_mm3, vol_mm3, vol_cm3


def build_mesh(mask, affine, smoothing_iters=10):
    """Marching Cubes -> mesh trimesh con vertici in coordinate FISICHE (mm).

    - measure.marching_cubes estrae la superficie isolivello 0.5 tra tumore(1)
      e sfondo(0), restituendo vertici in coordinate VOXEL (i, j, k).
    - apply_affine porta i vertici nello spazio fisico del paziente (mm), cosi'
      la mesh e' coerente col centro di massa calcolato in inference.py.
    """
    # level=0.5: confine tra le due classi. step_size=1 -> massimo dettaglio.
    verts_vox, faces, normals, _ = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, step_size=1
    )
    # Vertici voxel -> mm (spazio reale del paziente).
    verts_mm = apply_affine(affine, verts_vox)

    mesh = trimesh.Trimesh(vertices=verts_mm, faces=faces, process=True)

    # Smoothing leggero: i voxel danno una superficie "a gradini"; lo smoothing
    # Laplacian-Taubin la rende organica senza restringere il volume.
    if smoothing_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smoothing_iters)

    return mesh


def render_preview(mesh, com_world, out_png):
    """Rendering 3D statico della mesh con il centro di massa evidenziato."""
    fig = plt.figure(figsize=(9, 8))
    fig.patch.set_facecolor("#111111")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111111")

    coll = Poly3DCollection(
        mesh.vertices[mesh.faces],
        alpha=0.55, facecolor="#ff3b30", edgecolor="#5a0f0a", linewidths=0.05,
    )
    ax.add_collection3d(coll)

    # Centro di massa (target del robot).
    ax.scatter(*com_world, color="yellow", s=120, marker="+",
               linewidths=2.5, label="Centro di massa (target)")

    # Limiti = bounding box della mesh, assi in mm.
    mins = mesh.vertices.min(axis=0)
    maxs = mesh.vertices.max(axis=0)
    ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(mins[2], maxs[2])
    try:
        ax.set_box_aspect(maxs - mins)
    except Exception:
        pass

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color("white")
        axis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("Modello 3D del tumore rilevato", color="white", pad=12)
    ax.legend(loc="upper right", facecolor="#222222", labelcolor="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, facecolor="#111111")
    plt.close()


def main(mask_path, output_dir=config.OUTPUT_DIR, smoothing=10):
    os.makedirs(output_dir, exist_ok=True)

    seg = nib.load(mask_path)
    affine = seg.affine.astype(np.float64)
    mask = (seg.get_fdata() > 0.5).astype(np.uint8)

    if mask.sum() == 0:
        print("[3d] Maschera vuota: nessun tumore da modellare.")
        return None

    # Si passa l'affine: cosi' il filtro e' su un volume fisico (mm3) e non
    # sul semplice ordinamento delle componenti.
    mask = largest_component(mask, affine)
    if mask.sum() == 0:
        print('[3d] Nessuna componente sopra la soglia di volume.')
        return None

    # --- 1) VOLUME ---
    n_vox, voxel_vol, vol_mm3, vol_cm3 = compute_volume(mask, affine)
    # Diametro di una sfera equivalente (intuizione clinica sulla "grandezza").
    eq_diam_mm = 2.0 * (3.0 * vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)

    # --- 2) MESH 3D + export STL ---
    mesh = build_mesh(mask, affine, smoothing_iters=smoothing)
    com_world = mesh.center_mass if mesh.is_watertight else apply_affine(
        affine, np.array(ndimage.center_of_mass(mask))
    )

    base = os.path.splitext(os.path.splitext(os.path.basename(mask_path))[0])[0]
    stl_path = os.path.join(output_dir, f"{base}_mesh.stl")
    mesh.export(stl_path)

    # Dimensioni del bounding box (utili al planner per delimitare i raggi).
    bbox = mesh.bounds  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    bbox_dims = bbox[1] - bbox[0]

    # --- 3) PREVIEW PNG ---
    png_path = os.path.join(output_dir, f"{base}_mesh3d.png")
    render_preview(mesh, com_world, png_path)

    # --- REPORT ---
    print("\n" + "=" * 60)
    print("  MODELLO 3D DEL TUMORE - REPORT")
    print("=" * 60)
    print(f"  Voxel tumore           : {n_vox}")
    print(f"  Volume voxel singolo   : {voxel_vol:.4f} mm^3")
    print(f"  VOLUME TUMORE          : {vol_mm3:,.1f} mm^3  =  {vol_cm3:.2f} cm^3 (mL)")
    print(f"  Diametro sfera equiv.  : {eq_diam_mm:.1f} mm")
    print("-" * 60)
    print(f"  Mesh: {len(mesh.vertices):,} vertici, {len(mesh.faces):,} facce")
    print(f"  Watertight (chiusa)    : {mesh.is_watertight}")
    print(f"  Bounding box (mm)      : "
          f"{bbox_dims[0]:.1f} x {bbox_dims[1]:.1f} x {bbox_dims[2]:.1f}")
    print(f"  Centro di massa (mm)   : "
          f"X={com_world[0]:+.2f}  Y={com_world[1]:+.2f}  Z={com_world[2]:+.2f}")
    print("-" * 60)
    print(f"  STL salvato  : {stl_path}")
    print(f"  Preview PNG  : {png_path}")
    print("=" * 60 + "\n")

    return {
        "volume_mm3": vol_mm3,
        "volume_cm3": vol_cm3,
        "mesh": mesh,
        "stl_path": stl_path,
        "com_world_mm": com_world,
        "bbox_mm": bbox_dims,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Volume + mesh 3D (STL) del tumore segmentato."
    )
    parser.add_argument("--mask", required=True,
                        help="Maschera predetta .nii.gz (output di inference.py).")
    parser.add_argument("--out", default=config.OUTPUT_DIR)
    parser.add_argument("--smoothing", type=int, default=10,
                        help="Iterazioni di smoothing Taubin (0 = superficie a voxel).")
    args = parser.parse_args()

    assert os.path.exists(args.mask), f"Maschera non trovata: {args.mask}"
    main(args.mask, args.out, args.smoothing)
