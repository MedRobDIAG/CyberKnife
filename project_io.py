# ============================================================================
#  project_io.py - Salvataggio e apertura di un progetto (.ckproj).
#
#  PERCHE'
#  -------
#  Una seduta completa costa minuti di calcolo: inferenza e analisi di
#  incertezza (~25 s su GPU), modello 3D degli organi (~30 s), dosimetria
#  (~1 min), cinematica, confronto delle tecniche di moto (alcuni minuti).
#  Il progetto salva i RISULTATI di ogni fase, cosi' riaprirlo riporta la GUI
#  allo stesso punto senza rifare nulla.
#
#  IL FORMATO
#  ----------
#  Un unico file .ckproj, che e' un archivio npz compresso:
#    __meta__   JSON con la struttura dello stato e i riferimenti agli array
#    a0, a1...  gli array NumPy
#  Niente pickle, di proposito: un pickle aperto e' codice eseguito, e un
#  progetto e' un file che si scambia con altri. Gli oggetti del progetto
#  (fasci, mesh, basi del robot, configurazioni del respiro...) hanno un
#  codificatore esplicito qui sotto; un oggetto sconosciuto NON viene salvato
#  e finisce nell'elenco "omessi", che la GUI mostra invece di tacerlo.
#
#  COSA NON SI SALVA
#  -----------------
#  La TAC: e' il file di partenza e pesa centinaia di MB. Si salva il suo
#  percorso con dimensione e data, e all'apertura si ricarica (e si avvisa se
#  e' cambiata). Le grandezze che se ne ricavano in un secondo - la griglia
#  di densita', la mappa di dose interpolabile - si ricalcolano.
# ============================================================================
import json
import os
from datetime import datetime, timezone

import numpy as np

FORMAT_VERSION = 1
EXTENSION = ".ckproj"

#: Array float grandi con valori in [0, 1] (probabilita', incertezze): si
#: quantizzano a 16 bit, errore massimo 7.6e-6, un ordine di grandezza sotto
#: qualunque soglia usata nel progetto. Dimezza il file.
QUANT_MIN_SIZE = 1_000_000


# ---------------------------------------------------------------------------
#  Codifica
# ---------------------------------------------------------------------------
class _Encoder:
    def __init__(self):
        self.arrays = {}
        self.omitted = []

    def _store(self, a):
        key = f"a{len(self.arrays)}"
        a = np.asarray(a)
        if a.dtype == bool:
            self.arrays[key] = np.packbits(a.ravel())
            return {"__nd__": key, "bool": list(a.shape)}
        # Maschere salvate come uint8 0/1 (la maschera del tumore, i
        # riferimenti degli organi): a bit come le booleane, 8 volte meno.
        if (a.dtype == np.uint8 and a.size >= QUANT_MIN_SIZE
                and a.max(initial=0) <= 1):
            self.arrays[key] = np.packbits(a.ravel())
            return {"__nd__": key, "bool": list(a.shape), "u8": True}
        if (a.dtype.kind == "f" and a.size >= QUANT_MIN_SIZE
                and np.nanmin(a) >= 0.0 and np.nanmax(a) <= 1.0):
            self.arrays[key] = np.round(a * 65535.0).astype(np.uint16)
            return {"__nd__": key, "q16": str(a.dtype)}
        if a.dtype == object:
            raise TypeError("array di oggetti")
        self.arrays[key] = a
        return {"__nd__": key}

    def enc(self, obj, path="stato"):
        import kinematics as kin
        import motion_model as mm
        from dosimetry import Beam

        if obj is None or isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if isinstance(obj, float) and not np.isfinite(obj):
                return {"__float__": repr(obj)}
            return obj
        if isinstance(obj, np.generic):
            return self.enc(obj.item(), path)
        if isinstance(obj, np.ndarray):
            try:
                return self._store(obj)
            except TypeError:
                return {"__l__": [self.enc(v, f"{path}[{i}]")
                                  for i, v in enumerate(obj.tolist())]}
        if isinstance(obj, dict):
            return {"__d__": [[self.enc(k, path), self.enc(v, f"{path}.{k}")]
                              for k, v in obj.items()]}
        if isinstance(obj, list):
            return {"__l__": [self.enc(v, f"{path}[{i}]")
                              for i, v in enumerate(obj)]}
        if isinstance(obj, tuple):
            return {"__t__": [self.enc(v, f"{path}[{i}]")
                              for i, v in enumerate(obj)]}
        if isinstance(obj, Beam):
            return {"__beam__": {s: self.enc(getattr(obj, s), path)
                                 for s in Beam.__slots__}}
        if isinstance(obj, kin.RobotBase):
            return {"__base__": {"origin": self._store(obj.origin),
                                 "R": self._store(obj.R),
                                 "mount": self._store(obj.mount),
                                 "yaw": float(obj.yaw)}}
        if isinstance(obj, kin.CouchPose):
            return {"__couch__": obj.as_dict()}
        if isinstance(obj, kin.Box):
            return {"__box__": obj.as_dict()}
        if isinstance(obj, mm.MotionConfig):
            return {"__motion__": obj.as_dict()}
        if isinstance(obj, mm.BreathingTrace):
            return {"__trace__": {k: self.enc(getattr(obj, k), path)
                                  for k in ("t", "tumor", "surrogate",
                                            "inhaling", "cfg")}}
        if isinstance(obj, mm.TrackingResult):
            return {"__track__": {k: self.enc(getattr(obj, k), path)
                                  for k in ("t", "truth", "estimate",
                                            "active", "kv_t", "kv_err",
                                            "predictor")}}
        try:
            import trimesh
            if isinstance(obj, trimesh.Trimesh):
                return {"__mesh__": {
                    "v": self._store(np.asarray(obj.vertices, np.float32)),
                    "f": self._store(np.asarray(obj.faces, np.int32))}}
        except ImportError:
            pass
        self.omitted.append(f"{path} ({type(obj).__name__})")
        return None


class _Decoder:
    def __init__(self, arrays):
        self.arrays = arrays

    def _load(self, ref):
        a = self.arrays[ref["__nd__"]]
        if "bool" in ref:
            shape = tuple(ref["bool"])
            n = int(np.prod(shape))
            bits = np.unpackbits(a)[:n].reshape(shape)
            return bits if ref.get("u8") else bits.astype(bool)
        if "q16" in ref:
            return (a.astype(np.float32) / 65535.0).astype(ref["q16"])
        return a

    def dec(self, obj):
        import kinematics as kin
        import motion_model as mm
        from dosimetry import Beam

        if not isinstance(obj, (dict, list)):
            return obj
        if isinstance(obj, list):              # non dovrebbe capitare
            return [self.dec(v) for v in obj]
        if "__nd__" in obj:
            return self._load(obj)
        if "__float__" in obj:
            return float(obj["__float__"])
        if "__d__" in obj:
            return {self._key(k): self.dec(v) for k, v in obj["__d__"]}
        if "__l__" in obj:
            return [self.dec(v) for v in obj["__l__"]]
        if "__t__" in obj:
            return tuple(self.dec(v) for v in obj["__t__"])
        if "__beam__" in obj:
            d = {k: self.dec(v) for k, v in obj["__beam__"].items()}
            return Beam(d["source"], d["target"], d["cone_mm"],
                        weight=d["weight"], path_cm=d["path_cm"],
                        node_id=d["node_id"], target_id=d["target_id"],
                        bone_cm=d["bone_cm"])
        if "__base__" in obj:
            d = obj["__base__"]
            b = kin.RobotBase(mount=self._load(d["mount"]))
            b = b.relocated(self._load(d["origin"]), self._load(d["R"]))
            b.yaw = float(d["yaw"])
            return b
        if "__couch__" in obj:
            return kin.CouchPose.from_dict(obj["__couch__"])
        if "__box__" in obj:
            return kin.Box.from_dict(obj["__box__"])
        if "__motion__" in obj:
            return mm.MotionConfig.from_dict(obj["__motion__"])
        if "__trace__" in obj:
            d = {k: self.dec(v) for k, v in obj["__trace__"].items()}
            return mm.BreathingTrace(**d)
        if "__track__" in obj:
            d = {k: self.dec(v) for k, v in obj["__track__"].items()}
            return mm.TrackingResult(**d)
        if "__mesh__" in obj:
            import trimesh
            d = obj["__mesh__"]
            return trimesh.Trimesh(vertices=self._load(d["v"]),
                                   faces=self._load(d["f"]), process=False)
        return {k: self.dec(v) for k, v in obj.items()}

    def _key(self, k):
        k = self.dec(k)
        return tuple(k) if isinstance(k, list) else k


# ---------------------------------------------------------------------------
#  API
# ---------------------------------------------------------------------------
def ct_fingerprint(path):
    """Dimensione e data della TAC: per accorgersi se e' cambiata."""
    try:
        st = os.stat(path)
        return {"percorso": os.path.abspath(path), "byte": st.st_size,
                "modificato": st.st_mtime}
    except OSError:
        return {"percorso": path}


def save_project(path, state):
    """Scrive lo stato. Restituisce l'elenco degli oggetti non salvati.

    Scrittura atomica: si scrive accanto e si rinomina, cosi' un errore a
    meta' non distrugge il progetto precedente con lo stesso nome.
    """
    if not path.lower().endswith(EXTENSION):
        path += EXTENSION
    enc = _Encoder()
    tree = enc.enc(state)
    meta = {"formato": FORMAT_VERSION,
            "salvato": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "omessi": enc.omitted, "stato": tree}
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:            # file aperto: savez non aggiunge .npz
        np.savez_compressed(fh, __meta__=np.array(json.dumps(meta)),
                            **enc.arrays)
    os.replace(tmp, path)
    return path, enc.omitted


def load_project(path):
    """Rilegge uno stato. Restituisce (stato, metadati)."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["__meta__"]))
        if meta.get("formato") != FORMAT_VERSION:
            raise ValueError(
                f"Progetto in formato {meta.get('formato')}, atteso "
                f"{FORMAT_VERSION}.")
        arrays = {k: z[k] for k in z.files if k != "__meta__"}
    state = _Decoder(arrays).dec(meta["stato"])
    meta.pop("stato")
    return state, meta
