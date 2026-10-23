# ============================================================================
#  test_organ_mesh.py - Controlli sulla ricostruzione 3D di organi e scheletro.
#
#  Due livelli:
#
#  1. FANTOCCI SINTETICI. Una sfera di densita' nota ha superficie, volume e
#     curvatura calcolabili a mano: se la catena non ritrova il raggio entro
#     qualche decimo di millimetro, l'errore e' nella catena e non nei dati.
#     Sono anche i controlli che dimostrano le due correzioni piu' delicate:
#     il ritaglio simmetrico del campo (la posizione della superficie non deve
#     dipendere dal contrasto) e il riempimento dell'interno.
#
#  2. CASO REALE (lung_001). Soglie di qualita' per struttura, misurate contro
#     la superficie fisica ricavata dai valori Hounsfield, piu' i controlli
#     sulla segmentazione che la ricostruzione non puo' rimediare: le vie aeree
#     devono essere un albero connesso, il mediastino deve stare davanti alla
#     colonna.
#
#  USO:  python test_organ_mesh.py            (tutto)
#        python test_organ_mesh.py --sintetici (solo fantocci, pochi secondi)
# ============================================================================
import argparse
import os
import sys

import numpy as np
from scipy import ndimage

import organ_mesh as omsh

HERE = os.path.dirname(os.path.abspath(__file__))
CT = os.path.join(HERE, "Task06_Lung", "Task06_Lung", "imagesTr",
                  "lung_001.nii.gz")

_fails = []


def check(name, cond, extra=""):
    ok = bool(cond)
    if not ok:
        _fails.append(name)
    print(f"[{'OK ' if ok else 'FALLITO'}] {name} {extra}")
    return ok


# ---------------------------------------------------------------------------
def sphere_case(radius_mm=25.0, spacing=1.0, inside_hu=300.0,
                outside_hu=-1000.0, shape=80):
    """Sfera di densita' nota su griglia isotropa: (maschera, hu, affine)."""
    aff = np.eye(4)
    aff[:3, :3] = np.diag([spacing] * 3)
    aff[:3, 3] = -np.array([shape * spacing / 2.0] * 3)
    g = (np.indices((shape,) * 3).astype(np.float64)
         * spacing) - shape * spacing / 2.0
    r = np.sqrt((g ** 2).sum(axis=0))
    mask = r <= radius_mm
    hu = np.where(mask, inside_hu, outside_hu).astype(np.float32)
    return mask, hu, aff, r


def test_sintetici():
    print("\n--- 1) Fantoccio sferico: la catena ritrova il raggio? ---")
    R = 25.0
    mask, hu, aff, _ = sphere_case(R)
    rec = dict(omsh.RECIPES["bones"], level=0.0, faces=8000, band_mm=4.0,
               sigma_mm=0.6, smooth_iters=10, pass_band=0.1, min_ml=0.5)
    mesh = omsh.build_mesh(mask, hu, aff, rec)
    check("la sfera viene ricostruita", mesh is not None and len(mesh.faces) > 0)
    if mesh is None:
        return
    rr = np.linalg.norm(mesh.vertices, axis=1)
    check("raggio esatto entro 0.3 mm", abs(rr.mean() - R) < 0.3,
          f"{rr.mean():.3f} mm invece di {R}")
    check("superficie chiusa", mesh.is_watertight)
    v_att = 4.0 / 3.0 * np.pi * R ** 3
    err = 100.0 * (abs(mesh.volume) - v_att) / v_att
    check("volume entro il 2%", abs(err) < 2.0, f"{err:+.2f}%")
    check("una sola componente", mesh.body_count == 1, str(mesh.body_count))
    check("superficie liscia (sotto 12 gradi)",
          omsh.roughness_deg(mesh) < 12.0, f"{omsh.roughness_deg(mesh):.1f}d")

    # --- il contrasto NON deve spostare la superficie --------------------
    print("\n--- 1b) Il contrasto non sposta il contorno ---")
    # Era il difetto del ritiro: sfumare un gradino asimmetrico sposta lo zero
    # verso il lato di modulo minore. Con il ritaglio simmetrico due densita'
    # interne molto diverse devono dare lo stesso raggio.
    raggi = []
    for ins in (60.0, 1500.0):
        m2, hu2, aff2, _ = sphere_case(R, inside_hu=ins)
        mesh2 = omsh.build_mesh(m2, hu2, aff2, rec)
        raggi.append(float(np.linalg.norm(mesh2.vertices, axis=1).mean()))
    check("raggio indipendente dal contrasto (entro 0.1 mm)",
          abs(raggi[0] - raggi[1]) < 0.1,
          f"{raggi[0]:.3f} contro {raggi[1]:.3f} mm")

    # --- guscio cavo: l'interno va riempito -------------------------------
    print("\n--- 1c) Un guscio cavo diventa un solido ---")
    # E' il caso della costa: la soglia prende solo il corticale e la
    # superficie sarebbe a doppia parete. Con il riempimento deve restare un
    # guscio unico, e il volume quello della sfera piena.
    mask3, hu3, aff3, r3 = sphere_case(R)
    hu3 = np.where(r3 < R - 4.0, -1000.0, hu3).astype(np.float32)  # midollo
    mesh3 = omsh.build_mesh(mask3, hu3, aff3, rec)
    check("il guscio cavo da' una sola superficie",
          mesh3 is not None and mesh3.body_count == 1,
          str(None if mesh3 is None else mesh3.body_count))
    if mesh3 is not None and mesh3.is_watertight:
        err3 = 100.0 * (abs(mesh3.volume) - v_att) / v_att
        check("il volume e' quello della sfera piena (entro il 3%)",
              abs(err3) < 3.0, f"{err3:+.2f}%")
    senza = omsh.build_mesh(mask3, hu3, aff3, dict(rec, fill=False))
    check("senza riempimento la superficie si sdoppia davvero",
          senza is not None and senza.body_count > 1,
          f"{None if senza is None else senza.body_count} componenti")

    # --- distanza firmata --------------------------------------------------
    print("\n--- 1d) La metrica di distanza e' in millimetri ---")
    sdf = omsh.signed_distance_field(mask, aff)
    centro = np.array(mask.shape) // 2
    check("al centro la distanza e' -raggio",
          abs(abs(sdf[tuple(centro)]) - R) < 1.5,
          f"{sdf[tuple(centro)]:.2f} mm")
    pt = np.array([[R + 5.0, 0.0, 0.0]])
    d = float(omsh.sample_field(sdf, aff, pt)[0])
    check("a 5 mm fuori la distanza e' +5", abs(d - 5.0) < 0.6, f"{d:+.2f} mm")

    # --- la distanza firmata batte la gaussiana sulla maschera ------------
    print("\n--- 1e) Distanza firmata contro maschera sfumata ---")
    # Per le strutture senza appoggio nella densita' (mediastino, tumore) si
    # usa la distanza firmata: sfumare una maschera binaria tira le superfici
    # curve verso l'interno, e su una sfera l'effetto si misura.
    m_sdf = omsh.build_mesh(mask, None, aff,
                            dict(omsh.RECIPES["tumor"], faces=8000,
                                 sigma_mm=1.0))
    r_sdf = float(np.linalg.norm(m_sdf.vertices, axis=1).mean())
    from skimage import measure
    field = ndimage.gaussian_filter(mask.astype(np.float32), sigma=1.0)
    v, f, _, _ = measure.marching_cubes(field, level=0.5)
    r_gauss = float(np.linalg.norm(
        (aff[:3, :3] @ v.T + aff[:3, 3:4]).T, axis=1).mean())
    check("la distanza firmata sbaglia meno della maschera sfumata",
          abs(r_sdf - R) < abs(r_gauss - R),
          f"{r_sdf - R:+.3f} mm contro {r_gauss - R:+.3f} mm")


# ---------------------------------------------------------------------------
def torace_finto(spacing=2.0):
    """Fantoccio toracico: due polmoni, una trachea e una bolla gastrica.

    Serve a verificare le due regole che tengono in piedi la segmentazione su
    pazienti diversi, senza dipendere dal dataset:

      * il CONTROLLO DI ESPLOSIONE delle vie aeree: la trachea e' un tubo
        sottile collegato a un enorme volume d'aria (il polmone), e
        l'accrescimento deve fermarsi all'imbocco invece di divorarlo;
      * l'ADIACENZA per l'aria residua: la bolla gastrica e' aria staccata,
        lontana dai polmoni, e non deve diventare polmone.

    Restituisce (ct, affine, maschere_attese) con le maschere di verita'.
    """
    nx = ny = int(round(300.0 / spacing))
    nz = int(round(360.0 / spacing))
    aff = np.eye(4)
    aff[:3, :3] = np.diag([spacing] * 3)
    aff[:3, 3] = [-nx * spacing / 2.0, -ny * spacing / 2.0,
                  -nz * spacing / 2.0]
    x, y, z = np.indices((nx, ny, nz)).astype(np.float64) * spacing
    x -= nx * spacing / 2.0
    y -= ny * spacing / 2.0
    z -= nz * spacing / 2.0

    ct = np.full((nx, ny, nz), -1000.0, dtype=np.float32)
    # Tronco: cilindro di tessuto molle lungo l'asse testa-piedi.
    tronco = (x ** 2 / 130.0 ** 2 + y ** 2 / 95.0 ** 2) <= 1.0
    ct[tronco] = 40.0
    # Colonna, cosi' il vincolo "davanti alla colonna" ha un riferimento.
    ct[((x ** 2 + (y + 70.0) ** 2) <= 20.0 ** 2) & tronco] = 600.0
    # Polmoni: due ellissoidi d'aria, ben separati, nella metà craniale.
    for cx in (-60.0, 60.0):
        lung = (((x - cx) ** 2 / 50.0 ** 2 + y ** 2 / 65.0 ** 2
                 + (z - 70.0) ** 2 / 95.0 ** 2) <= 1.0)
        ct[lung] = -850.0
    # Trachea: tubo verticale in mezzo, dal bordo craniale fino ai polmoni.
    trachea = ((x ** 2 + (y - 10.0) ** 2) <= 9.0 ** 2) & (z > 60.0)
    ct[trachea] = -1000.0
    # Bolla gastrica: aria staccata, sotto i polmoni.
    stomaco = (((x - 40.0) ** 2 + (y - 30.0) ** 2
                + (z + 110.0) ** 2) <= 45.0 ** 2)
    ct[stomaco] = -1000.0
    return ct, aff, {"trachea": trachea, "stomaco": stomaco}


def test_fantoccio_segmentazione():
    import organs as og

    print("\n--- 1f) Fantoccio toracico: trachea, polmoni, bolla gastrica ---")
    sp = 2.0
    ct, aff, vero = torace_finto(sp)
    vox_ml = sp ** 3 / 1000.0
    avvisi = []
    masks, gaff, vols = og.segment_organs(
        ct, aff, sp, status_cb=lambda m: avvisi.append(m))

    v_trachea = float(vero["trachea"].sum()) * vox_ml
    v_stomaco = float(vero["stomaco"].sum()) * vox_ml
    aw = masks["airways"]
    lungs = masks["lung_r"] | masks["lung_l"]

    check("le vie aeree non divorano il polmone",
          vols["airways"] < 4.0 * v_trachea,
          f"{vols['airways']:.1f} cm3 contro {v_trachea:.1f} di trachea")
    check("le vie aeree trovano la trachea",
          float((aw & vero["trachea"]).sum()) * vox_ml > 0.4 * v_trachea,
          f"{100 * float((aw & vero['trachea']).sum()) / max(vero['trachea'].sum(), 1):.0f}% "
          f"della trachea")
    check("le vie aeree sono un pezzo unico",
          ndimage.label(aw)[1] == 1, f"{ndimage.label(aw)[1]} componenti")
    # L'ADIACENZA: la bolla gastrica e' aria, ma non e' polmone.
    dentro = float((lungs & vero["stomaco"]).sum()) * vox_ml
    check("la bolla gastrica non diventa polmone", dentro < 0.05 * v_stomaco,
          f"{dentro:.1f} cm3 su {v_stomaco:.1f}")
    check("la bolla gastrica non diventa una via aerea",
          float((aw & vero["stomaco"]).sum()) == 0)
    check("i due polmoni vengono separati",
          masks["lung_r"].any() and masks["lung_l"].any())
    check("l'esclusione dell'aria non polmonare viene dichiarata",
          any("aria non polmonare" in m for m in avvisi),
          str([m for m in avvisi if "aria" in m][:1]))

    # L'avviso di mediastino implausibile deve ESISTERE e scattare: da quella
    # maschera esce la dose al mediastino mostrata nel pannello, e su alcuni
    # pazienti la maschera ingloba fegato o milza (limite di metodo, non
    # parametro: alla quota del diaframma sono tessuto contiguo di pari
    # densita'). Si abbassa la soglia per provocarlo di proposito.
    limite = og.MEDIASTINUM_MAX_ML
    try:
        og.MEDIASTINUM_MAX_ML = 0.001
        avvisi2 = []
        og.segment_organs(ct, aff, sp, status_cb=lambda m: avvisi2.append(m))
    finally:
        og.MEDIASTINUM_MAX_ML = limite
    check("un mediastino implausibile viene dichiarato",
          any("mediastino" in m and "ATTENZIONE" in m for m in avvisi2))


# ---------------------------------------------------------------------------
#: Soglie di qualita' per struttura sul caso reale. Non sono aspirazioni: sono
#: i valori misurati con le ricette scelte da `--tune`, con un margine. Se un
#: giorno peggiorano, questo test lo dice.
#  I valori misurati su lung_001 sono: corpo 0.43 mm / 8.5 gradi, scheletro
#  0.65 / 35.5, polmoni 0.50 e 0.52 / 13.3 e 14.4, vie aeree 0.60 / 17.7,
#  mediastino 0.76 / 24.8. Le soglie stanno un po' sopra, per non fallire al
#  primo decimale, ma non tanto da lasciar passare una regressione vera.
#
#  La CHIUSURA si pretende dove ha senso: corpo, polmoni e mediastino sono
#  volumi compatti. Scheletro e vie aeree sono strutture sottili e ramificate,
#  con decine di componenti, e su quelle basta un lato scoperto su
#  venticinquemila facce: si controlla che ALMENO QUATTRO strutture su sei
#  siano chiuse, invece di pretenderlo da tutte e sei.
LIMITI = {
    "body": dict(fedelta=0.70, asprezza=12.0, schegge=1.0, chiusa=True,
                 volume=2.0),
    "bones": dict(fedelta=0.90, asprezza=40.0, schegge=1.5, chiusa=False),
    "lung_r": dict(fedelta=0.75, asprezza=17.0, schegge=1.0, chiusa=True,
                   volume=4.0),
    "lung_l": dict(fedelta=0.75, asprezza=18.0, schegge=1.0, chiusa=True,
                   volume=4.0),
    "airways": dict(fedelta=0.85, asprezza=22.0, schegge=1.0, chiusa=False),
    # Le "schegge" del mediastino sono per lo piu' triangoli allungati in zone
    # piatte, innocui: la metrica li conta comunque e su questa struttura
    # arriva al 6.6%. Cio' che conta - le superfici fuori posto - lo dice la
    # fedelta', ed e' 0.76 mm.
    "mediastinum": dict(fedelta=1.00, asprezza=28.0, schegge=9.0, chiusa=True,
                        volume=4.0),
}


def test_reale():
    import nibabel as nib
    import organs as og

    print("\n--- 2) Caso reale: segmentazione ---")
    img = nib.load(CT)
    ct = np.asanyarray(img.dataobj, dtype=np.float32)
    masks, gaff, vols = og.segment_organs(ct, img.affine, 2.0)

    # Le vie aeree nascono da un seme sulla trachea e crescono: devono essere
    # UN albero. Con il vecchio criterio ("l'aria che resta") si spezzavano in
    # sette frammenti.
    n_air = ndimage.label(masks["airways"])[1]
    check("le vie aeree sono un albero connesso", n_air == 1,
          f"{n_air} componenti, {vols['airways']:.0f} cm3")
    # Misurato su nove pazienti: 18.8-63.5 cm3, sempre in un pezzo unico. Il
    # tetto di AIRWAY_MAX_ML e' una rete di sicurezza: se lo si tocca, il
    # controllo di esplosione non ha funzionato e il programma lo dichiara.
    check("le vie aeree hanno un volume plausibile",
          10.0 <= vols["airways"] < og.AIRWAY_MAX_ML,
          f"{vols['airways']:.0f} cm3")

    # Il mediastino deve stare DAVANTI alla colonna: prima inghiottiva i
    # solchi paravertebrali e il canale vertebrale. Si usa LA STESSA funzione
    # della segmentazione, non una copia del criterio: la prima versione di
    # questo controllo ne riscriveva la logica a mano, dimenticava di
    # escludere lo sterno, e dichiarava 76985 voxel fuori posto che non lo
    # erano.
    ax = og._world_axis(img.affine, 0)
    ay = og._world_axis(img.affine, 1)
    front = og.anterior_to_spine(masks["body"], masks["bones"], ax, ay, 2.0,
                                 img.affine)
    dietro = int((masks["mediastinum"] & ~front).sum())
    check("il mediastino non sconfina dietro la colonna", dietro == 0,
          f"{dietro} voxel")
    # Il volume del mediastino NON e' un buon controllo: varia da 400 a 2600
    # cm3 fra pazienti perche' la maschera ingloba tessuto sotto-diaframmatico
    # dove cuore e fegato sono contigui. Si controllano invece le proprieta'
    # che devono valere sempre, e che il programma dichiari l'implausibilita'.
    check("il mediastino e' un pezzo unico",
          ndimage.label(masks["mediastinum"])[1] == 1,
          f"{ndimage.label(masks['mediastinum'])[1]} componenti")
    sopra = og.above_lung_floor(masks["lung_r"] | masks["lung_l"],
                                3 - ax - ay, img.affine, spacing_mm=2.0)
    sotto = int((masks["mediastinum"] & ~sopra).sum())
    check("il mediastino non scende sotto il diaframma", sotto == 0,
          f"{sotto} voxel")

    print("\n--- 3) Caso reale: qualita' delle mesh ---")
    built = omsh.build_all(masks, gaff, ct, img.affine)
    got = {o["key"]: o for o in built}
    check("tutte le strutture sono state ricostruite",
          set(LIMITI) <= set(got), str(sorted(set(LIMITI) - set(got))))
    for key, lim in LIMITI.items():
        o = got.get(key)
        if o is None:
            continue
        rec = omsh.RECIPES[key]
        hu = None
        if rec["source"] != "mask":
            hu, _, _ = omsh.organ_grid(ct, img.affine, masks[key], gaff,
                                       omsh.MESH_SPACING_MM)
        ref = omsh.reference_mask(o["riferimento"], hu, rec)
        rep = omsh.mesh_report(o["mesh"], ref, o["rif_affine"])
        print("   " + omsh.format_report(key, rep))
        check(f"{key}: fedelta' sotto {lim['fedelta']} mm",
              rep["media_mm"] <= lim["fedelta"], f"{rep['media_mm']:.2f} mm")
        check(f"{key}: asprezza sotto {lim['asprezza']} gradi",
              rep["asprezza_deg"] <= lim["asprezza"],
              f"{rep['asprezza_deg']:.1f}d")
        check(f"{key}: schegge sotto il {lim['schegge']}%",
              rep["schegge_pct"] <= lim["schegge"],
              f"{rep['schegge_pct']:.1f}%")
        check(f"{key}: la superficie non si gonfia ne' si ritira",
              abs(rep["bias_mm"]) <= 0.35, f"{rep['bias_mm']:+.2f} mm")
        if lim["chiusa"]:
            check(f"{key}: superficie chiusa", rep["chiusa"])
            if rep["chiusa"]:
                check(f"{key}: volume entro il {lim['volume']}%",
                      abs(rep["err_volume_pct"]) <= lim["volume"],
                      f"{rep['err_volume_pct']:+.1f}%")
    n_chiuse = sum(1 for k in LIMITI
                   if k in got and omsh.integrity(got[k]["mesh"])["chiusa"])
    check("almeno quattro strutture su sei sono chiuse", n_chiuse >= 4,
          f"{n_chiuse}/6")

    print("\n--- 4) La catena nuova batte quella storica ---")
    vecchie = {o["key"]: o["mesh"] for o in
               og.build_organ_meshes(masks, gaff)}
    meglio = 0
    for key in ("bones", "airways", "lung_r"):
        if key not in got or key not in vecchie:
            continue
        rec = omsh.RECIPES[key]
        hu = None
        if rec["source"] != "mask":
            hu, _, _ = omsh.organ_grid(ct, img.affine, masks[key], gaff,
                                       omsh.MESH_SPACING_MM)
        ref = omsh.reference_mask(got[key]["riferimento"], hu, rec)
        aff = got[key]["rif_affine"]
        sdf = omsh.signed_distance_field(ref, aff)
        a = omsh.surface_fidelity_mm(vecchie[key], ref, aff, sdf=sdf)
        b = omsh.surface_fidelity_mm(got[key]["mesh"], ref, aff, sdf=sdf)
        print(f"   {key:<10}storica {a['media_mm']:.2f} mm -> "
              f"nuova {b['media_mm']:.2f} mm")
        meglio += int(b["media_mm"] < a["media_mm"])
    check("la nuova catena e' piu' fedele su tutte le strutture provate",
          meglio == 3, f"{meglio}/3")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sintetici", action="store_true",
                    help="solo i fantocci, senza il caso reale")
    args = ap.parse_args()

    test_sintetici()
    test_fantoccio_segmentazione()
    if not args.sintetici:
        if os.path.exists(CT):
            test_reale()
        else:
            print(f"\nCaso reale saltato: {CT} non trovato")

    print("\n" + ("Tutti i controlli superati" if not _fails
                  else f"FALLITI: {_fails}"))
    sys.exit(1 if _fails else 0)
