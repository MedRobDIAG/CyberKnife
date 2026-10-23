# ============================================================================
#  export_scene.py - Esporta l'anatomia come mesh STL per il nodo ROS 2.
#
#  RViz sa disegnare mesh esterne (marker MESH_RESOURCE), quindi la scena
#  anatomica non va ricalcolata dentro ROS: la si segmenta una volta qui, con
#  la stessa pipeline della GUI (organs.py), e si scrivono gli STL dentro il
#  pacchetto.
#
#  Il nodo ROS non importa scipy, skimage o nibabel: in WSL c'e' solo rclpy e
#  numpy. Tutto il carico di imaging resta su Windows, dove gia' gira.
#
#  UNITA': gli STL sono scritti in METRI, l'unita' di ROS. La conversione dai
#  millimetri della CT avviene QUI e solo qui, cosi' il nodo non deve
#  ricordarsi di scalare le mesh (mentre per i fasci, che arrivano dal JSON in
#  mm, la conversione e' esplicita nel nodo).
#
#  USO:
#    python export_scene.py --ct <ct.nii.gz> --mask <tumore.nii.gz>
# ============================================================================
import argparse
import json
import os
import time

import numpy as np
import nibabel as nib
from scipy import ndimage

import kinematics as kin
import organs as og

MM_TO_M = 0.001

#: Organi da esportare. L'alfa e' quella di RViz: il corpo va tenuto molto
#: trasparente, altrimenti nasconde tutto cio' che sta dentro - che e'
#: esattamente quello che si vuole guardare.
SCENE_STYLE = {
    "body":        dict(color=(0.85, 0.72, 0.62), alpha=0.10, faces=40000),
    "bones":       dict(color=(0.93, 0.92, 0.88), alpha=0.55, faces=90000),
    "lung_r":      dict(color=(0.55, 0.75, 0.95), alpha=0.22, faces=25000),
    "lung_l":      dict(color=(0.55, 0.75, 0.95), alpha=0.22, faces=25000),
    "airways":     dict(color=(0.60, 0.90, 0.90), alpha=0.45, faces=12000),
    "mediastinum": dict(color=(0.90, 0.45, 0.45), alpha=0.30, faces=20000),
}


def _mesh_from_mask(mask, affine, faces_budget, sigma=0.9, smoothing=8,
                    hu=None, key=None):
    """Ricostruzione con la catena di organ_mesh (ricetta per struttura).

    Le mesh che finiscono in RViz sono le stesse che si vedono nel Tab 2: se
    qui si usasse un'altra catena, la scena della simulazione mostrerebbe
    un'anatomia diversa da quella su cui il piano e' stato calcolato. Il
    budget di facce e' invece piu' alto, perche' RViz rasterizza su GPU.
    """
    import organ_mesh as omsh
    rec = dict(omsh.RECIPES.get(key or "", omsh.RECIPES["mediastinum"]))
    if hu is None:
        rec["source"] = "mask"
    rec["faces"] = faces_budget
    return omsh.build_mesh(mask, hu, affine, rec)


def export_scene(ct_path, mask_path, out_dir, spacing_mm=2.0, verbose=True):
    def log(m):
        if verbose:
            print("  ", m)

    os.makedirs(out_dir, exist_ok=True)
    img = nib.load(ct_path)
    ct = np.asanyarray(img.dataobj, dtype=np.float32)

    masks, gaff, vols = og.segment_organs(ct, img.affine, spacing_mm,
                                          status_cb=log)

    # Stessa catena del Tab 2 (organ_mesh), con i budget di facce di RViz.
    import organ_mesh as omsh
    recipes = {k: dict(omsh.RECIPES[k], faces=SCENE_STYLE[k]["faces"])
               for k in SCENE_STYLE if k in omsh.RECIPES}
    built = omsh.build_all(masks, gaff, ct, img.affine, recipes=recipes,
                           keys=list(SCENE_STYLE), status_cb=log)

    entries = []
    for o in built:
        key, mesh = o["key"], o["mesh"]
        style = SCENE_STYLE[key]
        if mesh is None or len(mesh.faces) == 0:
            continue
        mesh.vertices = mesh.vertices * MM_TO_M          # mm -> m, una volta sola
        fn = f"{key}.stl"
        mesh.export(os.path.join(out_dir, fn))
        entries.append({"key": key, "file": fn, "color": list(style["color"]),
                        "alpha": style["alpha"], "faces": int(len(mesh.faces)),
                        "volume_cm3": round(float(vols.get(key, 0.0)), 1)})

    # --- Tumore: dalla maschera della Fase 1, non dalla segmentazione HU -----
    import postproc
    tum, n_les = postproc.primary_mask(
        nib.load(mask_path).get_fdata() > 0.5, img.affine)
    if n_les > 1:
        log(f"{n_les} lesioni rilevate: in scena va la principale")
    log("Mesh tumore...")
    # Ricetta "tumor": la maschera viene dalla rete, non da una soglia di
    # densita', quindi la superficie si ricava dalla sua distanza firmata. La
    # lesione e' piccola (~2.7 cm3) e la sua forma e' proprio cio' che il
    # piano conformale insegue: la ricetta liscia poco.
    tmesh = _mesh_from_mask(tum > 0, img.affine, 60000, key="tumor")
    if tmesh is not None:
        vox_ml = abs(np.linalg.det(np.asarray(img.affine)[:3, :3])) / 1000.0
        tmesh.vertices = tmesh.vertices * MM_TO_M
        tmesh.export(os.path.join(out_dir, "tumor.stl"))
        entries.append({"key": "tumor", "file": "tumor.stl",
                        "color": [1.0, 0.15, 0.15], "alpha": 0.95,
                        "faces": int(len(tmesh.faces)),
                        "volume_cm3": round(float(tum.sum()) * vox_ml, 2)})

    # Ostacoli per il modello di collisione. Vengono ricavati QUI, dalla stessa
    # maschera del corpo usata per le mesh, e scritti nel manifesto: il nodo
    # ROS li rilegge invece di ricalcolarli, cosi' la scatola che il planner
    # evita e quella che RViz disegna sono per costruzione la stessa. In
    # millimetri, come tutta la cinematica: la conversione in metri resta al
    # nodo, che gia' la fa per i fasci.
    bounds = kin.bounds_mm(masks["body"], gaff)
    obstacles = kin.treatment_obstacles(bounds)
    # Geometria del manipolatore che porta il lettino. Va nel manifesto perche'
    # dipende dal paziente (la colonna sta oltre il piede del lettino, la cui
    # quota viene dal dorso) e il nodo ROS non puo' ricalcolarla: non ha ne'
    # nibabel ne' le maschere.
    arm = kin.treatment_couch_arm(bounds)
    manifest = {"unita": "m", "unita_ostacoli": "mm",
                "sorgente_ct": ct_path, "sorgente_maschera": mask_path,
                "organi": entries,
                "ostacoli": [o.as_dict() for o in obstacles],
                "lettino": {
                    "colonna_mm": [round(float(v), 3) for v in arm.column],
                    "piano_a_riposo_mm": [round(float(v), 3) for v in arm.home],
                    "l1_mm": round(float(arm.l1), 1),
                    "l2_mm": round(float(arm.l2), 1),
                    "pavimento_mm": round(float(kin.floor_y_mm(bounds)), 1),
                }}
    with open(os.path.join(out_dir, "scene.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    return manifest


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Esporta l'anatomia in STL per RViz.")
    ap.add_argument("--ct", default="Task06_Lung/Task06_Lung/imagesTr/lung_001.nii.gz")
    ap.add_argument("--mask", default="outputs/lung_001_seg.nii.gz")
    ap.add_argument("--out", default=os.path.join(
        here, "ros2_ws", "src", "cyberknife_ros", "meshes"))
    ap.add_argument("--spacing", type=float, default=2.0)
    args = ap.parse_args()

    t0 = time.time()
    man = export_scene(args.ct, args.mask, args.out, args.spacing)
    print(f"\nScena esportata in {time.time() - t0:.1f}s -> {args.out}")
    print(f"{'organo':<14}{'facce':>9}{'volume':>12}")
    for e in man["organi"]:
        print(f"{e['key']:<14}{e['faces']:>9}{e['volume_cm3']:>10.1f} cm3")
    print()
    lc = man["lettino"]
    print(f"braccio del lettino: colonna {[round(v) for v in lc['colonna_mm']]}"
          f"  piano a riposo {[round(v) for v in lc['piano_a_riposo_mm']]}")
    print()
    print(f"{'ostacolo':<14}{'centro (mm)':>28}{'dimensioni (mm)':>28}")
    for o in man["ostacoli"]:
        c = [round(v) for v in o["centro_mm"]]
        d = [round(v) for v in o["dimensioni_mm"]]
        print(f"{o['nome']:<14}{str(c):>28}{str(d):>28}")
