# ============================================================================
#  kinematics.py - Cinematica del manipolatore CyberKnife (Fase 3).
#
#  MODELLO
#  -------
#  Il CyberKnife reale monta un acceleratore lineare da 6 MV sul polso di un
#  manipolatore seriale antropomorfo a 6 assi (KUKA KR 300 R2500 ultra):
#  tre giunti per la posizione, un POLSO SFERICO (tre assi concorrenti) per
#  l'orientamento. E' la struttura che rende risolvibile la cinematica inversa
#  in forma chiusa, senza iterazioni numeriche.
#
#  IL COMPITO E' A 5 GRADI DI LIBERTA', NON 6
#  ------------------------------------------
#  Il collimatore e' circolare: ruotare la testata attorno al proprio asse non
#  cambia il fascio erogato. Servono quindi 3 gradi per posizionare la sorgente
#  e 2 per orientarne l'asse, mentre il braccio ne offre 6. Il grado in eccesso
#  e' una RIDONDANZA: la si sfrutta cercando, fra le infinite rotazioni attorno
#  all'asse del fascio, quella che tiene i giunti piu' lontani dai fine corsa.
#
#  Questo modulo non importa ne' ROS ne' Qt: la stessa matematica gira
#  nell'applicazione Windows e dentro il nodo ROS 2 in WSL.
#
#  Convenzione: coordinate mondo in millimetri, sistema RAS del paziente
#  (+X destra, +Y anteriore, +Z craniale) - le stesse che escono dalla Fase 1.
# ============================================================================
import numpy as np

# ---------------------------------------------------------------------------
#  Geometria del manipolatore (millimetri e radianti)
#
# ---------------------------------------------------------------------------
#  Quote del KUKA KR 300 R2500 ultra, il manipolatore montato sul CyberKnife
#  M6/S7. Ricavate dall'URDF pubblico del pacchetto di supporto ROS e coerenti
#  con la portata dichiarata: A1 + A2 + D4 = 350 + 1150 + 1000 = 2500 mm, cioe'
#  proprio la "R2500" del nome.
D1 = 675.0     # base -> spalla (altezza)
A1 = 350.0     # scostamento orizzontale della spalla dall'asse di base
A2 = 1150.0    # braccio (spalla -> gomito)
A3 = 41.0      # scostamento del gomito
D4 = 1000.0    # avambraccio (gomito -> centro polso)
D6 = 240.0     # centro polso -> flangia

#: Distanza flangia -> punto focale della sorgente di raggi X, lungo l'asse
#: del fascio. E' la testata dell'acceleratore montata sulla flangia: un
#: linac X-band da 6 MV, compatto proprio per stare su un braccio. La quota
#: esatta non e' pubblicata dal costruttore, questa e' una stima coerente con
#: l'ingombro della testata; la distanza sorgente-isocentro (SAD = 800 mm) e'
#: invece un dato dichiarato, e la usa la dosimetria.
LINAC_LENGTH = 300.0

#: Distanza totale centro-polso -> sorgente. E' l'offset che va tolto per
#: ricavare il centro del polso dalla posa desiderata della sorgente.
WRIST_TO_SOURCE = D6 + LINAC_LENGTH

#: Portata nominale: dall'asse 1 al centro del polso a braccio disteso.
REACH_MM = A1 + A2 + np.hypot(A3, D4)

#: Limiti di giunto del KR 300 R2500 ultra (gradi), nell'ordine A1..A6.
JOINT_LIMITS_DEG = np.array([
    [-185.0, 185.0],
    [-140.0,  -5.0],
    [-120.0, 155.0],
    [-350.0, 350.0],
    [-122.5, 122.5],
    [-350.0, 350.0],
])
JOINT_LIMITS = np.deg2rad(JOINT_LIMITS_DEG)

#: Velocita' massime dei giunti (gradi/s), A1..A6. Valori nominali di
#: catalogo della famiglia KR QUANTEC ultra, arrotondati: servono a dire se
#: l'inseguimento del respiro chiede al braccio piu' di quanto possa dare,
#: non a descrivere il robot al decimo di grado.
JOINT_SPEED_LIMITS_DEG_S = np.array([105.0, 101.0, 107.0, 136.0, 129.0, 206.0])

JOINT_NAMES = ["a1", "a2", "a3", "a4", "a5", "a6"]


# ---------------------------------------------------------------------------
#  Utilita' di rotazione
# ---------------------------------------------------------------------------
def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def frame_from_axis(z_axis, roll=0.0):
    """Terna ortonormale con l'asse Z assegnato e rotazione libera attorno.

    E' qui che entra la ridondanza: `roll` parametrizza la famiglia di
    orientamenti che producono lo STESSO fascio, perche' il collimatore e'
    circolare e non distingue una rotazione attorno al proprio asse.
    """
    z = np.asarray(z_axis, dtype=np.float64)
    z = z / max(np.linalg.norm(z), 1e-12)
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(tmp, z))) > 0.95:
        tmp = np.array([1.0, 0.0, 0.0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return R @ rot_z(roll)


# ---------------------------------------------------------------------------
#  Posa del robot nella stanza
# ---------------------------------------------------------------------------
class RobotBase:
    """Collocazione della base del robot rispetto al paziente.

    Il paziente vive nelle coordinate della CT (RAS, millimetri). Il robot sta
    accanto al lettino: questa trasformazione porta dal sistema del robot a
    quello del paziente, ed e' l'unico punto in cui i due mondi si incontrano.

    ORIENTAMENTO DEL MONTAGGIO
    --------------------------
    Il robot e' ancorato al PAVIMENTO, quindi il suo asse A1 e' VERTICALE. Ma
    "verticale" non e' un asse della TAC: in convenzione RAS gli assi seguono
    il paziente, e per un paziente supino la verticale della stanza coincide
    con +Y (anteriore), mentre +Z e' l'asse testa-piedi, che nella stanza e'
    ORIZZONTALE, lungo il lettino.

    Ruotare la base attorno a Z del paziente - come farebbe un semplice
    rot_z(yaw) - descriverebbe quindi un robot coricato su un fianco, non uno
    appoggiato per terra. La rotazione di montaggio qui sotto porta gli assi
    del robot su quelli della stanza:

        asse X del robot -> +Z paziente  (lungo il lettino, verso la testa)
        asse Y del robot -> +X paziente  (verso la destra del paziente)
        asse Z del robot -> +Y paziente  (verticale della stanza)

    Il parametro `yaw_deg` ruota poi il robot attorno alla PROPRIA verticale,
    che e' il grado di liberta' che si ha davvero installandolo nella sala.
    """

    #: Porta la terna del robot (Z verticale) su quella del paziente (RAS).
    #: Le colonne sono le immagini di X, Y, Z del robot in coordinate paziente.
    MOUNT = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0]])

    #: Collocazione predefinita, misurata su un piano reale con lettino e
    #: paziente come ostacoli. Nella stanza: 800 mm alla sinistra del paziente,
    #: a pavimento, 150 mm verso i piedi rispetto all'isocentro.
    #:
    #: La distanza laterale NON e' libera. Lo sbraccio della spalla (A1 = 350
    #: mm, raggio d'ingombro 190) spazza un cerchio di 540 mm attorno all'asse
    #: della base ogni volta che il giunto A1 ruota: se la base sta piu' vicina
    #: di 540 mm dal bordo del lettino, quella parte lo attraversa qualunque
    #: sia la configurazione. Il bordo e' a X = -253, quindi serve X <= -793.
    #: Con X = -700 il braccio raggiungeva piu' fasci, ma NESSUN trasferimento
    #: era percorribile: 78 fasci su 121 restavano fuori.
    #:
    #: La quota Y NON e' libera: il robot appoggia sul pavimento, che sta
    #: floor_y_mm() sotto il dorso del paziente. Il valore qui e' quello del
    #: caso in esame (lung_001); chi carica un'altra immagine deve ricavarlo
    #: con floor_y_mm(bounds_mm(maschera_corpo, affine)), come fanno il Tab 3 e
    #: plan_export.py.
    #:
    #: Il vincolo di collisione cambia completamente la scelta rispetto alla
    #: sola cinematica: senza di esso sembravano ottime posizioni da cui il
    #: braccio, per raggiungere il bersaglio, attraversa il lettino.
    #:
    #: Lo yaw resta 0 e non e' un caso: si e' misurato che non cambia nulla,
    #: perche' il giunto A1 ha una corsa di +-185 gradi e assorbe da solo
    #: qualunque rotazione della base attorno alla verticale.
    def __init__(self, origin_mm=(-800.0, -830.0, -150.0), yaw_deg=0.0,
                 mount=None):
        self.origin = np.asarray(origin_mm, dtype=np.float64)
        self.yaw = np.deg2rad(yaw_deg)
        self.mount = self.MOUNT if mount is None else np.asarray(mount, float)
        # Prima si orienta il robot come e' montato, poi lo si gira attorno
        # alla sua verticale: l'ordine conta, invertirlo ruoterebbe attorno a
        # un asse del paziente invece che attorno al palo del robot.
        self.R = self.mount @ rot_z(self.yaw)

    def relocated(self, origin_mm, R):
        """Copia con posa imposta, senza ripassare da montaggio e yaw.

        Serve a chi calcola direttamente una trasformazione (per esempio
        l'effetto di una rotazione del lettino) e non puo' esprimerla come
        semplice yaw attorno alla verticale.
        """
        b = RobotBase(origin_mm=(0.0, 0.0, 0.0), yaw_deg=0.0, mount=self.mount)
        b.origin = np.asarray(origin_mm, dtype=np.float64)
        b.R = np.asarray(R, dtype=np.float64)
        return b

    def to_robot(self, p_world):
        """Punto dal sistema paziente a quello del robot."""
        return self.R.T @ (np.asarray(p_world, dtype=np.float64) - self.origin)

    def dir_to_robot(self, v_world):
        return self.R.T @ np.asarray(v_world, dtype=np.float64)

    def to_world(self, p_robot):
        return self.R @ np.asarray(p_robot, dtype=np.float64) + self.origin


# ---------------------------------------------------------------------------
#  Il lettino e' un SECONDO ROBOT (RoboCouch)
#
#  Nel CyberKnife il lettino non e' un piano inclinabile: e' montato
#  all'estremita' di un secondo manipolatore, che orienta il paziente con
#  continuita' per aiutare il braccio dell'acceleratore a raggiungere direzioni
#  che da fermo non prenderebbe. Il brevetto del sistema descrive la catena:
#
#      colonna verticale PRISMATICA  -> altezza del piano
#      spalla   ROTOIDALE (asse verticale) -> movimento nel piano orizzontale
#      gomito   ROTOIDALE (asse verticale) -> movimento nel piano orizzontale
#      polso    ROTOIDALE x3 (roll, pitch, yaw) -> orientamento del piano
#
#  cioe' una SCARA con polso RPY. La struttura e' comoda: la colonna fissa da
#  sola l'altezza, le due rotoidali in pianta la posizione, il polso
#  l'orientamento. La cinematica inversa si scrive in forma chiusa senza
#  iterazioni, esattamente come per il braccio dell'acceleratore.
#
#  IL TRUCCO CHE SEMPLIFICA TUTTO
#  ------------------------------
#  Fasci, tumore, organi e scatole di collisione vivono nel sistema della TAC e
#  sono SOLIDALI al piano del lettino: muovendolo si spostano insieme, quindi
#  in quel sistema non cambia nulla fra loro. L'unica cosa che si muove e' il
#  robot dell'acceleratore, avvitato al pavimento. Muovere il lettino con la
#  trasformazione T equivale percio' a lasciare tutto fermo e applicare T^-1
#  alla base del robot. Cosi' non serve ruotare gli ostacoli, che restano
#  scatole allineate agli assi, e il calcolo di distanza resta veloce.
#
#  CHE COSA PAGA E CHE COSA NO (misurato su 353 direzioni di fascio)
#  ----------------------------------------------------------------
#  lettino fermo                              83/353
#  migliore singolo orientamento             137/353
#  due orientamenti                          211/353
#  tre orientamenti                          238/353
#  sei orientamenti                          249/353   (satura)
#
#  Lo YAW e' la leva. Il ROLL del polso non paga (137 -> 144 da solo, 249 ->
#  254 in unione): resta nel modello perche' il meccanismo lo ha, ma non e' una
#  variabile di ottimizzazione. Le TRASLAZIONI non pagano affatto (83 -> fra 70
#  e 93): spostare il paziente non cambia le direzioni da cui lo si guarda, e
#  in qualche caso peggiora. Restano nel modello perche' il lettino le fa
#  davvero e servono all'allineamento, ma non le si cerca per raggiungibilita'.
#
#  IL PAZIENTE NON SI INCLINA. Il pitch e' un asse di CORREZIONE del polso,
#  come il roll: serve ad allineare il bersaglio, non a cercare fasci. Il
#  secondo braccio esiste per far RUOTARE il lettino, non per ribaltarlo, e un
#  paziente inclinato di 20 gradi non e' immobilizzabile in sicurezza.
#  Misurato con i filtri di meccanismo e di raggiungibilita' attivi, con tre
#  angoli e la base ottima:
#    lettino fermo                             80/353
#    tre yaw (griglia da 5) + pitch +-20      163/353
#    tre yaw (griglia da 5), senza inclinare  143/353
#    tre yaw (griglia da 13), senza inclinare 158/353
#  Rinunciare all'inclinazione costa quindi 5 direzioni su 353 (1.4%), e le si
#  recuperano quasi tutte infittendo la griglia di yaw - che non costa nulla in
#  comfort del paziente. Un quarto angolo di lettino ne aggiungerebbe altre 2
#  in cambio di un'altra manovra: non vale il tempo di trattamento.
# ---------------------------------------------------------------------------
#: Corse dei giunti del lettino. Lo yaw del piano e' ampio - e' il classico
#: "angolo di lettino" della radioterapia - mentre roll e pitch del polso sono
#: assi di correzione, con escursione limitata.
COUCH_YAW_RANGE_DEG = 95.0
COUCH_PITCH_RANGE_DEG = 20.0
COUCH_ROLL_RANGE_DEG = 10.0

#: Geometria del braccio che porta il lettino (mm).
COUCH_L1_MM = 800.0            # colonna -> gomito
COUCH_L2_MM = 700.0            # gomito -> centro del piano
COUCH_LIFT_RANGE_MM = (550.0, 1100.0)
#: Quanto la colonna sta oltre il piede del lettino, lungo l'asse del lettino.
COUCH_COLUMN_OFFSET_MM = 200.0

#: Limiti dei giunti rotoidali del lettino (gradi): spalla, gomito, roll,
#: pitch, yaw del piano.
COUCH_JOINT_LIMITS_DEG = np.array([
    [-170.0, 170.0],       # spalla
    [-155.0, 155.0],       # gomito
    [-COUCH_ROLL_RANGE_DEG, COUCH_ROLL_RANGE_DEG],
    [-COUCH_PITCH_RANGE_DEG, COUCH_PITCH_RANGE_DEG],
    [-180.0, 180.0],       # yaw del polso (si somma a spalla + gomito)
])
COUCH_JOINT_LIMITS = np.deg2rad(COUCH_JOINT_LIMITS_DEG)
COUCH_JOINT_NAMES = ["colonna", "spalla", "gomito", "roll", "pitch", "yaw"]

#: Raggi d'ingombro delle parti del lettino (mm): colonna, primo braccio,
#: secondo braccio. Il piano vero e proprio e' gia' fra gli ostacoli.
COUCH_LINK_RADII_MM = np.array([170.0, 150.0, 130.0])



class CouchPose:
    """Posa del piano del lettino: rototraslazione rigida del paziente.

    Gli angoli sono riferiti al paziente supino:
      yaw   -> attorno alla verticale della stanza (asse +Y del paziente)
      pitch -> attorno all'asse trasversale (+X), testa su / testa giu'
      roll  -> attorno all'asse del lettino (+Z)
    e `shift_mm` e' la traslazione del piano.

    Il perno delle rotazioni e' l'ISOCENTRO, non il centro del lettino: cosi'
    il bersaglio resta fermo mentre il paziente si orienta, che e' il
    comportamento voluto e quello che il lettino realizza compensando con le
    proprie traslazioni.
    """

    __slots__ = ("yaw", "pitch", "roll", "shift", "pivot", "R")

    def __init__(self, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                 shift_mm=(0.0, 0.0, 0.0), pivot_mm=(0.0, 0.0, 0.0)):
        # Il terzo argomento POSIZIONALE era il perno prima che il lettino
        # diventasse un meccanismo a sei gradi di liberta'. Un vettore che
        # arriva come roll e' una chiamata vecchia: meglio fermarsi subito che
        # applicare una rotazione senza senso e non accorgersene.
        if np.ndim(roll_deg) != 0:
            raise TypeError(
                "CouchPose: roll_deg deve essere uno scalare. Il perno va "
                "passato come pivot_mm=... (la firma e' cambiata quando il "
                "lettino e' diventato un manipolatore a 6 assi).")
        self.yaw = float(np.deg2rad(yaw_deg))
        self.pitch = float(np.deg2rad(pitch_deg))
        self.roll = float(np.deg2rad(roll_deg))
        self.shift = np.asarray(shift_mm, dtype=np.float64)
        self.pivot = np.asarray(pivot_mm, dtype=np.float64)
        self.R = rot_y(self.yaw) @ rot_x(self.pitch) @ rot_z(self.roll)

    @property
    def yaw_deg(self):
        return float(np.rad2deg(self.yaw))

    @property
    def pitch_deg(self):
        return float(np.rad2deg(self.pitch))

    @property
    def roll_deg(self):
        return float(np.rad2deg(self.roll))

    def is_identity(self):
        return (abs(self.yaw) < 1e-12 and abs(self.pitch) < 1e-12 and
                abs(self.roll) < 1e-12 and
                float(np.abs(self.shift).max()) < 1e-9)

    def apply(self, p):
        """Dove finisce un punto del paziente dopo il movimento del lettino."""
        p = np.asarray(p, dtype=np.float64)
        return (self.R @ (p - self.pivot).T).T + self.pivot + self.shift

    def base_in_patient(self, base):
        """La base del robot vista dal paziente, dopo il movimento del lettino.

        Il robot non si muove: si muove il paziente sotto di lui. Nel sistema
        del paziente questo si legge come la base che subisce la trasformazione
        inversa.
        """
        if self.is_identity():
            return base
        o = self.R.T @ (base.origin - self.pivot - self.shift) + self.pivot
        return base.relocated(o, self.R.T @ base.R)

    def as_dict(self):
        return {"yaw_deg": round(self.yaw_deg, 2),
                "pitch_deg": round(self.pitch_deg, 2),
                "roll_deg": round(self.roll_deg, 2),
                "shift_mm": [round(float(v), 3) for v in self.shift],
                "pivot_mm": [round(float(v), 3) for v in self.pivot]}

    @staticmethod
    def from_dict(d):
        if not d:
            return CouchPose()
        return CouchPose(yaw_deg=d.get("yaw_deg", 0.0),
                         pitch_deg=d.get("pitch_deg", 0.0),
                         roll_deg=d.get("roll_deg", 0.0),
                         shift_mm=d.get("shift_mm", (0.0, 0.0, 0.0)),
                         pivot_mm=d.get("pivot_mm", (0.0, 0.0, 0.0)))

    def __repr__(self):
        extra = ""
        if float(np.abs(self.shift).max()) > 1e-6:
            extra = f", shift {np.round(self.shift, 0)}"
        return (f"CouchPose(yaw={self.yaw_deg:.0f}, pitch={self.pitch_deg:.0f}"
                f", roll={self.roll_deg:.0f}{extra})")


class CouchArm:
    """Il manipolatore che porta il lettino: colonna + SCARA + polso RPY.

    Lavora nel sistema della STANZA (Z verticale), che e' quello naturale per
    una colonna verticale, e converte da e verso il sistema del paziente con la
    stessa matrice di montaggio del robot dell'acceleratore. Una sola costante
    governa entrambi.
    """

    def __init__(self, tabletop_home_mm, floor_y_mm, couch_length_mm=None,
                 l1=COUCH_L1_MM, l2=COUCH_L2_MM):
        M = RobotBase.MOUNT                      # stanza -> paziente
        self.M = M
        self.l1, self.l2 = float(l1), float(l2)
        self.home = np.asarray(tabletop_home_mm, dtype=np.float64)
        length = COUCH_LENGTH_MM if couch_length_mm is None else couch_length_mm
        # La colonna sta oltre il piede del lettino, sul pavimento: e' il punto
        # in cui il secondo robot e' ancorato, e non si muove mai.
        self.column = np.array([self.home[0], float(floor_y_mm),
                                self.home[2] - length / 2.0
                                - COUCH_COLUMN_OFFSET_MM])
        self.col_room = M.T @ self.column
        self._home_room = M.T @ self.home

    # -- conversioni -----------------------------------------------------
    def to_room(self, p):
        return self.M.T @ np.asarray(p, dtype=np.float64)

    def to_patient(self, p):
        return self.M @ np.asarray(p, dtype=np.float64)

    # -- cinematica ------------------------------------------------------
    def forward(self, q):
        """Dai giunti alla posa del piano: (punto, orientamento) nel paziente.

        Il polso sta SOTTO il piano di COUCH_ARM_DROP_MM: il braccio sostiene
        il lettino da sotto, come il RoboCouch vero. Prima i due bracci in
        pianta stavano alla quota del centro del piano e, con 150 mm di raggio,
        il link attraversava il paziente.
        """
        h, s, e, r, pt, y = (float(v) for v in q)
        base = self.col_room + np.array([0.0, 0.0, h])
        elbow = base + self.l1 * np.array([np.cos(s), np.sin(s), 0.0])
        wrist = elbow + self.l2 * np.array([np.cos(s + e), np.sin(s + e), 0.0])
        R_room = rot_z(s + e) @ rot_x(r) @ rot_y(pt) @ rot_z(y)
        # Il sostegno e' solidale al polso: ruotando il lettino ruota con lui.
        tool = wrist + R_room @ np.array([0.0, 0.0, COUCH_ARM_DROP_MM])
        return self.M @ tool, self.M @ R_room @ self.M.T

    def link_points(self, q):
        """Punti della catena del lettino, in coordinate paziente.

        Sono gli estremi delle capsule d'ingombro: piede della colonna, cima
        della colonna, gomito, POLSO - non il centro del piano, che sta
        COUCH_ARM_DROP_MM piu' in alto ed e' gia' coperto dalla scatola del
        lettino fra gli ostacoli.
        """
        h, s, e = float(q[0]), float(q[1]), float(q[2])
        foot = self.col_room
        top = foot + np.array([0.0, 0.0, h])
        elbow = top + self.l1 * np.array([np.cos(s), np.sin(s), 0.0])
        wrist = elbow + self.l2 * np.array([np.cos(s + e), np.sin(s + e), 0.0])
        return (np.array([foot, top, elbow, wrist]) @ self.M.T)

    def inverse(self, p_patient, R_patient):
        """Giunti che portano il piano nella posa data. None se non ci arriva.

        La catena si disaccoppia: l'altezza viene dalla colonna, la posizione
        in pianta dalle due rotoidali (triangolo a due lati, come una SCARA),
        l'orientamento dal polso. Nessuna iterazione.
        """
        p = self.to_room(p_patient)
        R = self.M.T @ np.asarray(R_patient, dtype=np.float64) @ self.M
        # Dal piano si scende al POLSO: e' quello che la catena deve
        # raggiungere, il piano ci sta sopra appeso al sostegno.
        p = p - R @ np.array([0.0, 0.0, COUCH_ARM_DROP_MM])

        h = float(p[2] - self.col_room[2])
        if not (COUCH_LIFT_RANGE_MM[0] <= h <= COUCH_LIFT_RANGE_MM[1]):
            return None

        dx, dy = float(p[0] - self.col_room[0]), float(p[1] - self.col_room[1])
        d = float(np.hypot(dx, dy))
        if d > self.l1 + self.l2 or d < abs(self.l1 - self.l2):
            return None
        cos_e = np.clip((d * d - self.l1 ** 2 - self.l2 ** 2)
                        / (2.0 * self.l1 * self.l2), -1.0, 1.0)

        best, best_margin = None, -np.inf
        for sign in (+1.0, -1.0):
            e = sign * float(np.arccos(cos_e))
            s = np.arctan2(dy, dx) - np.arctan2(self.l2 * np.sin(e),
                                                self.l1 + self.l2 * np.cos(e))
            s = _wrap(s)
            # Polso: Rx(roll) Ry(pitch) Rz(yaw) e' cio' che resta togliendo la
            # rotazione accumulata in pianta. L'estrazione e' diretta.
            W = rot_z(s + e).T @ R
            pitch = float(np.arcsin(np.clip(W[0, 2], -1.0, 1.0)))
            if abs(np.cos(pitch)) < 1e-8:
                continue                      # polso in singolarita'
            roll = float(np.arctan2(-W[1, 2], W[2, 2]))
            yaw = float(np.arctan2(-W[0, 1], W[0, 0]))
            q = np.array([h, s, e, roll, pitch, yaw])
            ang = q[1:]
            # Tolleranza sul fine corsa. Gli orientamenti che l'ottimizzatore
            # prova cadono spesso ESATTAMENTE sul limite (pitch 20 gradi su una
            # corsa di 20): senza margine venivano rifiutati per l'errore
            # nell'ultimo bit, e il piano finiva per contenere angoli che poi
            # il controllo dichiarava irrealizzabili.
            eps = 1e-9
            if np.any(ang < COUCH_JOINT_LIMITS[:, 0] - eps) or \
               np.any(ang > COUCH_JOINT_LIMITS[:, 1] + eps):
                continue
            ang = np.clip(ang, COUCH_JOINT_LIMITS[:, 0],
                          COUCH_JOINT_LIMITS[:, 1])
            q[1:] = ang
            half = (COUCH_JOINT_LIMITS[:, 1] - COUCH_JOINT_LIMITS[:, 0]) / 2.0
            ctr = (COUCH_JOINT_LIMITS[:, 1] + COUCH_JOINT_LIMITS[:, 0]) / 2.0
            m = float(np.min(1.0 - np.abs(ang - ctr) / half))
            if m > best_margin:
                best, best_margin = q, m
        return best

    def solve_pose(self, couch):
        """Giunti che realizzano una CouchPose. None se il lettino non ci arriva.

        E' il filtro che rende le pose FISICHE: senza, l'ottimizzatore
        proporrebbe orientamenti che nessun meccanismo puo' assumere.
        """
        p = couch.apply(self.home)
        return self.inverse(p, couch.R)

    def home_joints(self):
        return self.inverse(self.home, np.eye(3))


# ---------------------------------------------------------------------------
#  Cinematica diretta
# ---------------------------------------------------------------------------
def forward_kinematics(q, base=None):
    """Dai sei angoli alla posa della sorgente.

    Restituisce (posizione_sorgente, direzione_fascio, centro_polso), nel
    sistema del paziente se `base` e' fornita, altrimenti in quello del robot.
    """
    q = np.asarray(q, dtype=np.float64)
    q1, q2, q3, q4, q5, q6 = q

    # --- Catena posizionale: base, spalla, gomito ---
    # Il piano del braccio ruota con q1; dentro quel piano, spalla e gomito
    # sono una catena planare a due segmenti.
    c1, s1 = np.cos(q1), np.sin(q1)
    r = A1 + A2 * np.cos(q2) + np.hypot(A3, D4) * np.cos(q2 + q3 + _elbow_phi())
    z = D1 - A2 * np.sin(q2) - np.hypot(A3, D4) * np.sin(q2 + q3 + _elbow_phi())
    wrist = np.array([r * c1, r * s1, z])

    # --- Orientamento: R03 dalla catena posizionale, poi il polso sferico ---
    R03 = rot_z(q1) @ rot_y(q2 + q3 + _elbow_phi())
    R36 = rot_z(q4) @ rot_y(q5) @ rot_z(q6)
    R06 = R03 @ R36

    beam_dir = R06 @ np.array([0.0, 0.0, 1.0])
    source = wrist + WRIST_TO_SOURCE * beam_dir

    if base is not None:
        return (base.to_world(source), base.R @ beam_dir, base.to_world(wrist))
    return source, beam_dir, wrist


def _elbow_phi():
    """Angolo costante fra l'asse del gomito e la congiungente col polso.

    L'avambraccio ha uno scostamento A3, quindi la retta gomito-polso non e'
    allineata all'asse del giunto: e' una costante geometrica, non un
    parametro, e va sottratta ovunque si ragioni su quel segmento.
    """
    return np.arctan2(A3, D4)


# ---------------------------------------------------------------------------
#  Ingombro del braccio e collisioni
#
#  Un fascio dosimetricamente ottimo puo' richiedere una posa che porta il
#  braccio DENTRO il lettino o dentro il paziente. La cinematica da sola non se
#  ne accorge: risolve gli angoli e li dichiara validi. Serve quindi un modello
#  di ingombro, per quanto grossolano.
#
#  Il braccio e' approssimato con quattro CAPSULE (segmento + raggio): colonna,
#  braccio, avambraccio, testata. E' il compromesso usuale nei planner: una
#  capsula racchiude un link cilindrico con pochi parametri, e la distanza
#  punto-segmento e' analitica. Gli ostacoli sono scatole allineate agli assi
#  del paziente, che e' esattamente la forma del lettino e una copertura
#  ragionevole del torace.
#
#  I raggi sono volutamente GENEROSI rispetto alla meccanica reale: un modello
#  di collisione che sottostima l'ingombro e' peggio che inutile, perche'
#  autorizza pose che nella sala colpirebbero il paziente.
# ---------------------------------------------------------------------------
#: Raggio d'ingombro di ciascuna capsula, in millimetri.
#:
#: La catena base->spalla e' spezzata in DUE capsule perche' il braccio fa una
#: L: colonna verticale alta D1, poi sbraccio orizzontale A1. Una sola capsula
#: dritta dalla base alla spalla lascerebbe scoperto lo spigolo della L, che
#: dista 311 mm dall'asse di quella retta - piu' del raggio - cioe' proprio la
#: parte piu' voluminosa del robot non sarebbe controllata.
LINK_RADII_MM = np.array([200.0,   # colonna base -> spigolo
                          190.0,   # sbraccio spigolo -> spalla
                          170.0,   # braccio spalla -> gomito
                          140.0,   # avambraccio gomito -> polso
                          175.0])  # polso -> sorgente (testata acceleratore)
N_CAPSULES = len(LINK_RADII_MM)

#: Quanti punti si campionano su ogni capsula per la distanza dalle scatole.
#: La distanza segmento-scatola non ha forma chiusa comoda; campionare e'
#: robusto e, con il raggio come margine, piu' che sufficiente qui.
_CAPSULE_SAMPLES = 9


class Box:
    """Ostacolo: parallelepipedo allineato agli assi del paziente (mm)."""

    __slots__ = ("lo", "hi", "name")

    def __init__(self, center, size, name="ostacolo"):
        c = np.asarray(center, dtype=np.float64)
        h = np.asarray(size, dtype=np.float64) / 2.0
        self.lo, self.hi, self.name = c - h, c + h, name

    def distance(self, pts):
        """Distanza (mm) di ciascun punto dalla scatola; 0 se dentro."""
        p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        d = np.maximum(np.maximum(self.lo - p, 0.0), p - self.hi)
        return np.linalg.norm(d, axis=1)

    def as_dict(self):
        c = (self.lo + self.hi) / 2.0
        return {"nome": self.name, "centro_mm": c.tolist(),
                "dimensioni_mm": (self.hi - self.lo).tolist()}

    @staticmethod
    def from_dict(d):
        return Box(d["centro_mm"], d["dimensioni_mm"], d.get("nome", "ostacolo"))


def link_points(q, base=None):
    """Punti della catena: base, spigolo, spalla, gomito, polso, sorgente.

    Sono gli estremi delle capsule. Si ricavano dalla stessa geometria della
    cinematica diretta, non da una copia: se i due divergessero il modello di
    collisione controllerebbe un braccio che non esiste.
    """
    q = np.asarray(q, dtype=np.float64)
    q1, q2 = q[0], q[1]
    c1, s1 = np.cos(q1), np.sin(q1)
    l3, phi = np.hypot(A3, D4), _elbow_phi()

    knee = np.array([0.0, 0.0, D1])                   # spigolo della L
    shoulder = np.array([A1 * c1, A1 * s1, D1])
    r_e = A1 + A2 * np.cos(q2)
    z_e = D1 - A2 * np.sin(q2)
    elbow = np.array([r_e * c1, r_e * s1, z_e])

    src, beam_dir, wrist = forward_kinematics(q)      # sistema del robot
    pts = np.array([[0.0, 0.0, 0.0], knee, shoulder, elbow, wrist, src])
    if base is not None:
        pts = (base.R @ pts.T).T + base.origin
    return pts


def _capsule_samples(pts):
    """Punti campionati lungo le quattro capsule, con il rispettivo raggio."""
    t = np.linspace(0.0, 1.0, _CAPSULE_SAMPLES).reshape(-1, 1)
    seg = []
    rad = []
    for k in range(N_CAPSULES):
        a, b = pts[k], pts[k + 1]
        seg.append(a + (b - a) * t)
        rad.append(np.full(_CAPSULE_SAMPLES, LINK_RADII_MM[k]))
    return np.vstack(seg), np.concatenate(rad)


def link_points_batch(Q, base=None):
    """link_points per N configurazioni in una volta sola. (N,6,3) in mm.

    Il controllo di collisione lungo un trasferimento valuta decine di
    configurazioni: farlo una alla volta paga l'interprete Python molte piu'
    volte del calcolo vero. Qui la stessa geometria e' scritta in forma
    vettoriale. Un test verifica che coincida con link_points.
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    q1, q2, q3, q4, q5, q6 = (Q[:, i] for i in range(6))
    c1, s1 = np.cos(q1), np.sin(q1)
    l3, phi = np.hypot(A3, D4), _elbow_phi()
    a = q2 + q3 + phi

    r_s = np.full_like(q1, A1)
    r_e = A1 + A2 * np.cos(q2)
    z_e = D1 - A2 * np.sin(q2)
    r_w = r_e + l3 * np.cos(a)
    z_w = z_e - l3 * np.sin(a)

    # Direzione del fascio: Rz(q1) Ry(a) Rz(q4) Ry(q5) Rz(q6) applicata a Z.
    # Rz(q6) non muove Z, quindi non compare.
    bx = np.cos(q4) * np.sin(q5)
    by = np.sin(q4) * np.sin(q5)
    bz = np.cos(q5)
    ux = bx * np.cos(a) + bz * np.sin(a)
    uz = -bx * np.sin(a) + bz * np.cos(a)
    dx = ux * c1 - by * s1
    dy = ux * s1 + by * c1
    dz = uz

    zero = np.zeros_like(q1)
    d1 = np.full_like(q1, D1)
    pts = np.stack([
        np.stack([zero, zero, zero], axis=-1),
        np.stack([zero, zero, d1], axis=-1),               # spigolo della L
        np.stack([r_s * c1, r_s * s1, d1], axis=-1),
        np.stack([r_e * c1, r_e * s1, z_e], axis=-1),
        np.stack([r_w * c1, r_w * s1, z_w], axis=-1),
        np.stack([r_w * c1 + WRIST_TO_SOURCE * dx,
                  r_w * s1 + WRIST_TO_SOURCE * dy,
                  z_w + WRIST_TO_SOURCE * dz], axis=-1),
    ], axis=1)
    if base is not None:
        pts = pts @ base.R.T + base.origin
    return pts


#: Raggi delle capsule ripetuti sui punti campionati, precalcolati una volta.
_SAMPLE_RADII = np.repeat(LINK_RADII_MM, _CAPSULE_SAMPLES)
_SAMPLE_T = np.linspace(0.0, 1.0, _CAPSULE_SAMPLES).reshape(1, -1, 1)


def collides_batch(Q, base, obstacles):
    """Vettore booleano: quali configurazioni portano il braccio in un ostacolo."""
    if not obstacles:
        return np.zeros(len(np.atleast_2d(Q)), dtype=bool)
    pts = link_points_batch(Q, base)                       # (N,6,3)
    a = pts[:, :N_CAPSULES, :][:, :, None, :]              # inizio capsule
    b = pts[:, 1:, :][:, :, None, :]                       # fine capsule
    samp = (a + (b - a) * _SAMPLE_T[None]).reshape(len(pts), -1, 3)
    out = np.zeros(len(pts), dtype=bool)
    for box in obstacles:
        d = np.maximum(np.maximum(box.lo - samp, 0.0), samp - box.hi)
        out |= np.any(np.linalg.norm(d, axis=-1) < _SAMPLE_RADII, axis=1)
    return out


def collides(q, base, obstacles):
    """True se la posa porta una parte del braccio dentro un ostacolo."""
    if not obstacles:
        return False
    pts, rad = _capsule_samples(link_points(q, base))
    for box in obstacles:
        if np.any(box.distance(pts) < rad):
            return True
    return False


def clearance_mm(q, base, obstacles):
    """Distanza minima fra l'ingombro del braccio e gli ostacoli (mm).

    Negativa in caso di compenetrazione. Serve a preferire, fra due pose
    entrambe valide, quella che passa piu' larga.
    """
    if not obstacles:
        return np.inf
    pts, rad = _capsule_samples(link_points(q, base))
    return float(min(np.min(box.distance(pts) - rad) for box in obstacles))


#: Passo angolare del campionamento lungo un trasferimento (gradi).
#: A braccio esteso (~2.4 m) un grado sposta la punta di circa 42 mm, meno
#: dello spessore del lettino (70 mm): con passi piu' larghi il controllo puo'
#: scavalcare l'ostacolo senza vederlo. Misurato: a 12 campioni fissi sfuggiva
#: un trasferimento che a 24 risultava in collisione.
PATH_STEP_DEG = 1.0


def _capsule_points_robot(Q):
    """Punti campionati sulle capsule, in coordinate ROBOT (base esclusa).

    Tenerli nel sistema del robot permette di applicare poi una base DIVERSA a
    ogni campione, che e' cio' che serve quando il lettino si muove insieme al
    braccio: la base effettiva cambia lungo il percorso.
    """
    pts = link_points_batch(Q, None)                       # (N, 6, 3)
    a = pts[:, :N_CAPSULES, :][:, :, None, :]
    b = pts[:, 1:, :][:, :, None, :]
    return (a + (b - a) * _SAMPLE_T[None]).reshape(len(pts), -1, 3)


def collides_moving(Q, Rs, Os, obstacles):
    """Collisioni con una base DIVERSA per ogni configurazione.

    Serve ai trasferimenti in cui il lettino ruota mentre il braccio si sposta:
    il paziente gira sotto il robot, quindi nel suo sistema la base si muove.
    Controllare solo gli estremi non basta - la compenetrazione, se c'e',
    avviene nel mezzo.
    """
    if not obstacles:
        return np.zeros(len(np.atleast_2d(Q)), dtype=bool)
    loc = _capsule_points_robot(Q)                         # (N, S, 3)
    # Una rotazione e una traslazione per campione.
    samp = np.einsum("nij,nsj->nsi", Rs, loc) + Os[:, None, :]
    out = np.zeros(len(samp), dtype=bool)
    for box in obstacles:
        d = np.maximum(np.maximum(box.lo - samp, 0.0), samp - box.hi)
        out |= np.any(np.linalg.norm(d, axis=-1) < _SAMPLE_RADII, axis=1)
    return out


def transfer_collides(q0, q1, base, obstacles, couch0=None, couch1=None,
                      n=None):
    """Trasferimento con lettino e braccio in movimento SIMULTANEO.

    E' il movimento che rende fluida la simulazione e che il lettino
    robotizzato fa davvero: non "fermati, ruota il paziente, riparti" ma le due
    catene che si muovono insieme. Il controllo interpola entrambe e verifica
    ogni passo.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if couch0 is None and couch1 is None:
        return path_collides(q0, q1, base, obstacles, n=n)
    if n is None:
        span = float(np.rad2deg(np.abs(q1 - q0).max()))
        if couch0 is not None and couch1 is not None:
            span = max(span, abs(couch1.yaw_deg - couch0.yaw_deg),
                       abs(couch1.pitch_deg - couch0.pitch_deg),
                       abs(couch1.roll_deg - couch0.roll_deg))
        n = int(max(12, min(240, np.ceil(span / PATH_STEP_DEG) + 1)))
    c0 = couch0 if couch0 is not None else CouchPose(pivot_mm=couch1.pivot)
    c1 = couch1 if couch1 is not None else CouchPose(pivot_mm=couch0.pivot)
    u = np.linspace(0.0, 1.0, n)
    Q = q0 + (q1 - q0) * u.reshape(-1, 1)
    Rs = np.empty((n, 3, 3))
    Os = np.empty((n, 3))
    for k, t in enumerate(u):
        c = CouchPose(c0.yaw_deg + (c1.yaw_deg - c0.yaw_deg) * t,
                      c0.pitch_deg + (c1.pitch_deg - c0.pitch_deg) * t,
                      c0.roll_deg + (c1.roll_deg - c0.roll_deg) * t,
                      c0.shift + (c1.shift - c0.shift) * t, c0.pivot)
        b = c.base_in_patient(base)
        Rs[k], Os[k] = b.R, b.origin
    return bool(collides_moving(Q, Rs, Os, obstacles).any())


def couch_arm_obstacles(arm, couch, exclude_below=None):
    """L'ingombro del BRACCIO DEL LETTINO come ostacoli per l'acceleratore.

    Le due catene condividono la stanza e possono urtarsi. Le parti sotto il
    piano stanno gia' dentro il volume proibito del lettino, quindi qui
    interessa quello che sporge: si generano scatole allineate lungo i segmenti
    della catena, che e' un'approssimazione grossolana ma conservativa.
    """
    q = arm.solve_pose(couch)
    if q is None:
        return []
    pts = arm.link_points(q)
    out = []
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        if exclude_below is not None and max(a[1], b[1]) < exclude_below:
            continue            # interamente sotto il piano: gia' coperto
        c = (a + b) / 2.0
        half = np.abs(b - a) / 2.0 + COUCH_LINK_RADII_MM[k]
        out.append(Box(c, 2.0 * half, name=f"lettino_braccio_{k}"))
    return out


def path_collides(q0, q1, base, obstacles, n=None):
    """Collisione lungo il TRASFERIMENTO fra due pose.

    Controllare solo le pose di erogazione non basta: il braccio le raggiunge
    muovendosi, e la strada fra due configurazioni entrambe libere puo'
    attraversare il lettino. E' il caso classico che si vede solo in
    simulazione, quando ormai il braccio e' passato dentro il paziente.
    """
    if not obstacles:
        return False
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if n is None:
        span = float(np.rad2deg(np.abs(q1 - q0).max()))
        n = int(max(12, min(240, np.ceil(span / PATH_STEP_DEG) + 1)))
    u = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return bool(collides_batch(q0 + (q1 - q0) * u, base, obstacles).any())


#: Dimensioni del lettino di trattamento (mm): larghezza, spessore, lunghezza.
#: Il PIANO del lettino non e' un parametro: coincide con il dorso del
#: paziente, che sul lettino appoggia. Ricavarlo dall'anatomia invece di
#: fissarlo a mano evita il lettino "a mezz'aria" e tiene d'accordo il vincolo
#: dosimetrico della Fase 2 con il modello di collisione della Fase 3.
COUCH_WIDTH_MM = 520.0
COUCH_THICKNESS_MM = 70.0
COUCH_LENGTH_MM = 2200.0

#: Quanto il braccio del lettino corre SOTTO il piano (mm).
#:
#: Prima era zero: i due bracci in pianta stavano sulla stessa quota del
#: CENTRO del piano, e con un raggio d'ingombro di 150 mm il cilindro
#: sbucava 115 mm sopra il dorso del paziente - in RViz il link attraversava
#: il corpo. Nel RoboCouch vero il braccio sta sotto il piano e lo sostiene
#: da sotto.
#:
#: La quota si ricava dalla geometria, non a occhio: meta' spessore del piano,
#: piu' il raggio del braccio piu' grosso, piu' un franco di 10 mm. Cosi' la
#: generatrice superiore del braccio resta 10 mm sotto la faccia inferiore del
#: piano.
COUCH_ARM_DROP_MM = (COUCH_THICKNESS_MM / 2.0
                     + float(COUCH_LINK_RADII_MM[1:].max()) + 10.0)

#: Altezza del piano del lettino da terra. E' questa a fissare la quota del
#: PAVIMENTO, non il contrario: il lettino appoggia sul dorso del paziente, che
#: viene dall'immagine, e il pavimento sta 850 mm piu' sotto - l'altezza di
#: trattamento tipica di un lettino robotizzato.
COUCH_FLOOR_MM = 850.0

#: Il volume SOTTO il lettino e' precluso. Il piano e' spesso appena 70 mm, e
#: senza il sostegno il braccio ci passa sotto: geometricamente lecito, ma in
#: sala il lettino ha una colonna e un basamento, e nessuno fa passare
#: l'acceleratore sotto il paziente. Si modella quindi l'intero volume fra
#: pavimento e piano come occupato.
COUCH_PEDESTAL = True

#: Margine aggiunto attorno al paziente. La TAC inquadra solo il torace, ma il
#: paziente ha anche testa e gambe: la scatola si estende per tutta la
#: lunghezza del lettino, altrimenti il braccio passerebbe tranquillamente
#: attraverso le parti che l'immagine non riprende.
PATIENT_MARGIN_MM = 25.0


def bounds_mm(mask, affine):
    """Estremi in millimetri (RAS) di una maschera booleana. Solo NumPy."""
    idx = np.array(np.nonzero(np.asarray(mask)), dtype=np.float64)
    if idx.size == 0:
        return None
    A = np.asarray(affine, dtype=np.float64)
    world = A[:3, :3] @ idx + A[:3, 3:4]
    return np.array([world.min(axis=1), world.max(axis=1)])


def floor_y_mm(body_bounds):
    """Quota del pavimento (mm, RAS), ricavata dall'altezza del lettino.

    Il robot ci appoggia sopra: e' il valore che deve avere la Y della base.
    """
    return float(np.asarray(body_bounds)[0, 1]) - COUCH_FLOOR_MM


def tabletop_home_mm(body_bounds):
    """Centro del piano del lettino a riposo (mm, RAS).

    Sta sotto il dorso del paziente, che sul lettino appoggia: la quota viene
    dall'immagine, non da un parametro.
    """
    lo, hi = np.asarray(body_bounds, dtype=np.float64)
    return np.array([(lo[0] + hi[0]) / 2.0,
                     lo[1] - COUCH_THICKNESS_MM / 2.0,
                     (lo[2] + hi[2]) / 2.0])


def treatment_couch_arm(body_bounds):
    """Il manipolatore che porta il lettino, dalla geometria del paziente.

    Costruito dagli stessi estremi del corpo da cui escono gli ostacoli: piano
    e braccio non possono percio' descrivere due lettini diversi.
    """
    return CouchArm(tabletop_home_mm(body_bounds), floor_y_mm(body_bounds))


def treatment_obstacles(body_bounds):
    """Lettino e paziente come scatole, a partire dagli estremi del corpo.

    `body_bounds` e' la matrice 2x3 restituita da bounds_mm sulla maschera del
    corpo. Restituisce la lista di Box da passare a solve_ik / solve_plan.
    """
    lo, hi = np.asarray(body_bounds, dtype=np.float64)
    cx = (lo[0] + hi[0]) / 2.0          # asse sinistra-destra del paziente
    cz = (lo[2] + hi[2]) / 2.0          # asse testa-piedi
    back_y = lo[1]                      # dorso: qui appoggia il lettino

    if COUCH_PEDESTAL:
        # Dal pavimento fino al piano: un unico volume proibito.
        h = COUCH_FLOOR_MM
        couch = Box(center=(cx, back_y - h / 2.0, cz),
                    size=(COUCH_WIDTH_MM, h, COUCH_LENGTH_MM), name="lettino")
    else:
        couch = Box(center=(cx, back_y - COUCH_THICKNESS_MM / 2.0, cz),
                    size=(COUCH_WIDTH_MM, COUCH_THICKNESS_MM, COUCH_LENGTH_MM),
                    name="lettino")

    m = PATIENT_MARGIN_MM
    patient = Box(
        center=(cx, (back_y + hi[1]) / 2.0, cz),
        size=(hi[0] - lo[0] + 2 * m, hi[1] - back_y + 2 * m, COUCH_LENGTH_MM),
        name="paziente")
    return [couch, patient]


# ---------------------------------------------------------------------------
#  Cinematica inversa
# ---------------------------------------------------------------------------
def _ik_position(wrist_r):
    """Giunti 1-3 dal centro del polso. Restituisce tutte le soluzioni valide.

    Il polso sferico DISACCOPPIA il problema: il suo centro dipende solo dai
    primi tre giunti, quindi la posizione si risolve senza sapere nulla
    dell'orientamento. Senza questa proprieta' servirebbe una soluzione
    numerica a 6 incognite.
    """
    x, y, z = wrist_r
    sols = []
    l3 = np.hypot(A3, D4)          # lunghezza effettiva dell'avambraccio
    phi = _elbow_phi()

    for q1 in (np.arctan2(y, x), np.arctan2(y, x) + np.pi):
        # Raggio nel piano del braccio: negativo se il braccio punta
        # all'indietro (configurazione "spalla ribaltata").
        r = np.hypot(x, y) * np.sign(np.cos(q1) * x + np.sin(q1) * y)
        dr, dz = r - A1, z - D1
        d = np.hypot(dr, dz)
        if d > A2 + l3 or d < abs(A2 - l3):
            continue               # fuori portata su questo ramo

        # Teorema del coseno sul triangolo spalla-gomito-polso.
        cos_e = np.clip((d * d - A2 * A2 - l3 * l3) / (2 * A2 * l3), -1.0, 1.0)
        for sign in (+1.0, -1.0):          # gomito alto / gomito basso
            e = sign * np.arccos(cos_e)
            k1 = A2 + l3 * np.cos(e)
            k2 = l3 * np.sin(e)
            # Catena planare a due segmenti: la FK compone
            #   dr = A2 cos(q2) + l3 cos(q2+e)
            #   dz = -[A2 sin(q2) + l3 sin(q2+e)]
            # da cui q2 = atan2(-dz, dr) - atan2(k2, k1), e poiche'
            # atan2(-y,x) = -atan2(y,x) i due termini si SOTTRAGGONO entrambi.
            q2 = -(np.arctan2(dz, dr) + np.arctan2(k2, k1))
            q3 = e - phi
            sols.append(np.array([_wrap(q1), _wrap(q2), _wrap(q3)]))
    return sols


def _ik_orientation(q123, R_target):
    """Giunti 4-6 dalla rotazione residua (angoli di Eulero ZYZ).

    Il polso realizza R36 = Rz(q4) Ry(q5) Rz(q6): estrarre gli angoli e'
    l'estrazione ZYZ classica. Le due soluzioni corrispondono a q5 positivo o
    negativo, cioe' al "polso dritto" e al "polso ribaltato".
    """
    q1, q2, q3 = q123
    R03 = rot_z(q1) @ rot_y(q2 + q3 + _elbow_phi())
    R36 = R03.T @ R_target
    out = []
    for sign in (+1.0, -1.0):
        q5 = sign * np.arccos(np.clip(R36[2, 2], -1.0, 1.0))
        if abs(np.sin(q5)) < 1e-8:
            # Singolarita' di polso: q4 e q6 diventano ridondanti fra loro,
            # si fissa q4 e si assorbe tutto su q6.
            q4 = 0.0
            q6 = np.arctan2(R36[1, 0], R36[0, 0])
        else:
            q4 = np.arctan2(R36[1, 2] * sign, R36[0, 2] * sign)
            q6 = np.arctan2(R36[2, 1] * sign, -R36[2, 0] * sign)
        out.append(np.array([_wrap(q4), _wrap(q5), _wrap(q6)]))
    return out


def _wrap(a):
    """Riporta un angolo in (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def within_limits(q):
    q = np.asarray(q, dtype=np.float64)
    return bool(np.all(q >= JOINT_LIMITS[:, 0]) and np.all(q <= JOINT_LIMITS[:, 1]))


def limit_margin(q):
    """Distanza minima dai fine corsa, normalizzata (1 = a meta' corsa).

    Serve a scegliere fra piu' soluzioni valide: una posa a un grado dal fine
    corsa e' formalmente valida ma inutilizzabile appena il piano cambia.
    """
    q = np.asarray(q, dtype=np.float64)
    lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    half = (hi - lo) / 2.0
    center = (hi + lo) / 2.0
    return float(np.min(1.0 - np.abs(q - center) / half))


#: Soglia sotto la quale una posa e' considerata "troppo vicina al fine corsa".
#: Non e' un vincolo rigido: e' il punto in cui la barriera inizia a mordere.
MARGIN_FLOOR = 0.15
#: Peso della barriera, in radianti-equivalenti di movimento: quanto extra
#: movimento si e' disposti a pagare pur di stare lontani dai fine corsa.
#: Misurato sui 112 fasci del piano reale, dalla base predefinita:
#:      peso  6 -> margine minimo 0.040, 12 pose sotto la soglia
#:      peso 25 -> margine minimo 0.109,  0 pose sotto la soglia
#: Il costo e' modesto (2720 -> 3045 gradi di movimento complessivo, +12%).
MARGIN_BARRIER = 25.0


#: Franco minimo desiderato dagli ostacoli (mm). Sopra questa soglia la
#: distanza non conta piu': fra due pose entrambe larghe si torna a scegliere
#: per movimento e margine di giunto.
CLEARANCE_TARGET_MM = 120.0
#: Peso del franco nel punteggio, in radianti-equivalenti di movimento.
CLEARANCE_WEIGHT = 4.0


def solve_ik(source_world, beam_dir_world, base, n_roll=24, q_ref=None,
             margin_floor=MARGIN_FLOOR, margin_barrier=MARGIN_BARRIER,
             obstacles=None):
    """Cinematica inversa per un fascio.

    Cerca, fra le rotazioni attorno all'asse del fascio (la ridondanza) e fra
    i rami di gomito/spalla/polso, la configurazione ammissibile migliore.

    Criterio di scelta: se `q_ref` e' dato (la posa del fascio precedente) si
    minimizza il movimento dei giunti, perche' fra un fascio e l'altro il robot
    deve fisicamente spostarsi e il tempo di trattamento e' dominato da quei
    trasferimenti. Altrimenti si massimizza il margine dai fine corsa.

    Il solo movimento pero' non basta come criterio: la soluzione piu' economica
    e' spesso quella appoggiata al fine corsa di A2, che ha appena 135 gradi di
    corsa. Una posa formalmente valida ma a un grado dal limite non ha alcun
    margine di manovra, quindi al costo di movimento si somma una BARRIERA che
    diventa proibitiva sotto `margin_floor`. Non e' un vincolo rigido: se non
    esiste alternativa la posa viene comunque accettata, perche' perdere un
    fascio e' peggio che erogarlo da una configurazione scomoda.

    Con `obstacles` le pose che portano il braccio dentro il lettino o dentro
    il paziente vengono SCARTATE, non penalizzate: una compenetrazione non e'
    una posa scomoda, e' una posa che non si puo' eseguire. Fra quelle libere
    si preferisce comunque chi passa piu' larga, fino a `CLEARANCE_TARGET_MM`.

    Restituisce (q, info) con q=None se il fascio non e' raggiungibile.
    """
    p_s = base.to_robot(source_world)
    n = base.dir_to_robot(beam_dir_world)
    n = n / max(np.linalg.norm(n), 1e-12)

    # Il centro del polso non dipende dall'orientamento attorno all'asse:
    # si ricava una volta sola, arretrando dalla sorgente lungo il fascio.
    wrist = p_s - WRIST_TO_SOURCE * n

    best, best_score, best_cl = None, -np.inf, np.inf
    n_pos_sols = n_in_limits = n_collide = 0
    for q123 in _ik_position(wrist):
        n_pos_sols += 1
        for roll in np.linspace(0.0, 2 * np.pi, n_roll, endpoint=False):
            R_target = frame_from_axis(n, roll)
            for q456 in _ik_orientation(q123, R_target):
                q = _resolve_turns(np.concatenate([q123, q456]), q_ref,
                                   margin_floor, margin_barrier)
                if q is None or not within_limits(q):
                    continue
                n_in_limits += 1
                cl = np.inf
                if obstacles:
                    cl = clearance_mm(q, base, obstacles)
                    if cl < 0.0:
                        n_collide += 1
                        continue
                m = limit_margin(q)
                if q_ref is None:
                    score = m
                else:
                    # Movimento pesato: i giunti di base muovono molta massa e
                    # sono i piu' lenti, quindi contano di piu' del polso.
                    # Niente wrapping: dopo _resolve_turns la differenza E' il
                    # movimento fisico, e avvolgerla nasconderebbe proprio i
                    # percorsi che escono dai fine corsa.
                    w = np.array([1.0, 1.0, 1.0, 0.4, 0.4, 0.2])
                    score = -float(np.sum(w * np.abs(q - q_ref)))
                    if m < margin_floor:
                        # Barriera quadratica: trascurabile appena sopra la
                        # soglia, dominante quando il margine va a zero.
                        d = (margin_floor - m) / margin_floor
                        score -= margin_barrier * d * d
                if obstacles and np.isfinite(cl):
                    # Bonus saturante: sotto il franco desiderato la vicinanza
                    # pesa, sopra non cambia piu' nulla.
                    score += CLEARANCE_WEIGHT * min(cl, CLEARANCE_TARGET_MM)                         / CLEARANCE_TARGET_MM
                if score > best_score:
                    best, best_score, best_cl = q, score, cl

    info = {"rami_posizione": n_pos_sols,
            "raggiungibile": best is not None,
            "entro_i_limiti": n_in_limits,
            "scartate_per_collisione": n_collide,
            "franco_mm": (float(best_cl) if best is not None and
                          np.isfinite(best_cl) else None),
            "margine_limiti": limit_margin(best) if best is not None else 0.0}
    return best, info


def _r06(q):
    """Orientamento della flangia nel sistema del robot (come in FK)."""
    q1, q2, q3, q4, q5, q6 = np.asarray(q, dtype=np.float64)
    return (rot_z(q1) @ rot_y(q2 + q3 + _elbow_phi())
            @ rot_z(q4) @ rot_y(q5) @ rot_z(q6))


def solve_ik_follow(source_world, q_ref, base, obstacles=None):
    """Cinematica inversa per INSEGUIRE: stessa orientazione di `q_ref`,
    sorgente spostata in `source_world`.

    L'inseguimento del respiro trasla il fascio senza ruotarlo, quindi la
    posa da raggiungere e' completamente nota - anche la rotazione attorno
    all'asse, che solve_ik tratta come ridondanza libera. Lasciarla libera
    era un errore misurato: con poche rotazioni provate il solutore ne
    sceglieva una diversa da quella del piano, e passarci chiedeva ad A6
    migliaia di gradi al secondo (fascio sospeso nel 50% dei passi anche con
    il tumore fermo). Fra i rami si prende il piu' vicino a `q_ref`.

    Restituisce q oppure None.
    """
    R = _r06(q_ref)
    n = R @ np.array([0.0, 0.0, 1.0])
    wrist = base.to_robot(source_world) - WRIST_TO_SOURCE * n
    w = np.array([1.0, 1.0, 1.0, 0.4, 0.4, 0.2])
    best, best_cost = None, np.inf
    for q123 in _ik_position(wrist):
        for q456 in _ik_orientation(q123, R):
            q = _resolve_turns(np.concatenate([q123, q456]), q_ref)
            if q is None or not within_limits(q):
                continue
            cost = float(np.sum(w * np.abs(q - q_ref)))
            if cost < best_cost:
                best, best_cost = q, cost
    if best is not None and obstacles and collides(best, base, obstacles):
        return None
    return best


def _wrap_vec(v):
    return (np.asarray(v) + np.pi) % (2 * np.pi) - np.pi


#: Determinazioni ammissibili di un angolo di giunto.
#:
#: La cinematica inversa restituisce angoli riportati in (-pi, pi], ma un
#: giunto con corsa piu' ampia di un giro - A4 e A6 arrivano a +-350 gradi -
#: puo' raggiungere la stessa posa con determinazioni diverse (q, q+2pi, ...).
#: La scelta NON e' indifferente: fra due pose consecutive il braccio percorre
#: fisicamente la differenza fra le determinazioni scelte, e la strada piu'
#: corta in senso ciclico puo' passare fuori dai fine corsa.
#:
#: Esempio concreto misurato su questo piano: A1 (+-185 gradi) passava da 170 a
#: -170 gradi con uno scarto ciclico di soli 20 gradi, ma quel percorso
#: attraversa 190 gradi, che il giunto non puo' raggiungere. Sommando questi
#: passaggi la traiettoria continua arrivava a 219.8 gradi, ben oltre il
#: limite, pur essendo ogni singola posa formalmente valida.
def _determinations(q, j):
    """Tutti i valori q + 2*pi*k che cadono entro i limiti del giunto j."""
    lo, hi = JOINT_LIMITS[j]
    out = []
    k = int(np.floor((lo - q) / (2.0 * np.pi)))
    while True:
        c = q + 2.0 * np.pi * k
        if c > hi + 1e-12:
            break
        if c >= lo - 1e-12:
            out.append(c)
        k += 1
    return out


def _joint_margin(value, j):
    """Margine dai fine corsa del solo giunto j (1 = a meta' corsa)."""
    lo, hi = JOINT_LIMITS[j]
    return 1.0 - abs(value - (hi + lo) / 2.0) / ((hi - lo) / 2.0)


def _resolve_turns(q, q_ref=None, margin_floor=None, margin_barrier=None):
    """Sceglie, giunto per giunto, la determinazione ammissibile migliore.

    Con `q_ref` si privilegia la piu' vicina alla posa precedente, cosi' il
    movimento e' quello davvero compiuto e resta dentro la corsa: fra due
    angoli entrambi ammissibili il giunto si muove in modo monotono e non puo'
    uscire dall'intervallo. Senza `q_ref` si sceglie la piu' centrata.

    Alla sola distanza si somma pero' una BARRIERA sul margine. Sceglierla solo
    per vicinanza e' un errore misurabile: se la determinazione piu' vicina
    cade a ridosso del fine corsa, quella a un giro di distanza - spesso
    perfettamente comoda - non veniva mai presa in considerazione, perche' la
    scelta avveniva PRIMA del punteggio e il resto del solutore non poteva piu'
    rimediare. Con la base a 800 mm il giunto A1 finiva cosi' a 184 gradi su
    185 di corsa, e alzare la barriera generale non serviva a niente.

    Restituisce None se anche un solo giunto non ha determinazioni ammissibili:
    quella posa e' irraggiungibile.
    """
    floor = MARGIN_FLOOR if margin_floor is None else margin_floor
    weight = MARGIN_BARRIER if margin_barrier is None else margin_barrier
    out = np.empty(6, dtype=np.float64)
    for j in range(6):
        cand = _determinations(float(q[j]), j)
        if not cand:
            return None
        if q_ref is None:
            center = float(np.mean(JOINT_LIMITS[j]))
            out[j] = min(cand, key=lambda c: abs(c - center))
        else:
            ref = float(q_ref[j])

            def _cost(c):
                cost = abs(c - ref)
                m = _joint_margin(c, j)
                if m < floor:
                    d = (floor - m) / floor
                    cost += weight * d * d
                return cost

            out[j] = min(cand, key=_cost)
    return out


def find_safe_shell(base, obstacles, n1=37, n23=31, n_best=5):
    """Configurazioni RITIRATE da cui ruotare la base in sicurezza.

    Servono come vie di transito. Con lettino e paziente di mezzo capita spesso
    che due pose entrambe eseguibili non siano collegabili in linea retta nello
    spazio dei giunti: la strada diretta passa dentro il lettino. Quello che
    funziona e' il movimento reale di un braccio: RITIRARSI, RUOTARE sulla base
    e RISCENDERE.

    Se ne restituiscono PIU' D'UNA, ordinate per franco decrescente. Il
    candidato migliore in media non e' sempre percorribile per una coppia
    specifica di pose: misurato, con il lettino ruotato di 90 gradi il guscio
    migliore lasciava appena 14 mm e ogni transito falliva, mentre alternative
    leggermente peggiori in media funzionavano. Provarne qualcuna costa poco e
    recupera fasci che altrimenti si perdono.

    Restituisce una lista di (q2, q3, franco_mediano_mm).
    """
    lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    q1s = np.linspace(lo[0], hi[0], n1)
    out = []
    for q2 in np.linspace(lo[1], hi[1], n23):
        for q3 in np.linspace(lo[2], hi[2], n23):
            Q = np.zeros((len(q1s), 6))
            Q[:, 0], Q[:, 1], Q[:, 2] = q1s, q2, q3
            if not within_limits(Q[0]):
                continue
            cl = np.array([clearance_mm(q, base, obstacles) for q in Q])
            out.append((float(q2), float(q3), float(np.median(cl))))
    out.sort(key=lambda r: -r[2])
    # Si scartano i gusci quasi identici: due (q2, q3) vicini danno la stessa
    # manovra, e proverebbero la stessa strada gia' fallita.
    picked = []
    for cand in out:
        if all(abs(cand[0] - p[0]) + abs(cand[1] - p[1]) > 0.25
               for p in picked):
            picked.append(cand)
        if len(picked) >= n_best:
            break
    return picked


def transit_route(q_from, q_to, shell, base, obstacles):
    """Percorso ritira-ruota-riscendi fra due pose. None se non praticabile.

    Restituisce i due punti di passaggio intermedi. Vengono verificati TUTTI E
    TRE i tratti, rotazione centrale compresa: con il lettino modellato fino a
    terra la rotazione a braccio ritirato non e' piu' sicura per costruzione.
    """
    if not shell:
        return None
    # Piu' gusci candidati: si prova il primo che porta a destinazione.
    if isinstance(shell[0], (list, tuple, np.ndarray)):
        for cand in shell:
            r = transit_route(q_from, q_to, tuple(cand[:2]), base, obstacles)
            if r is not None:
                return r
        return None
    q2, q3 = shell[0], shell[1]
    # q5 = 0 nei punti di passaggio: con il polso dritto la testata prosegue
    # lungo l'asse del braccio e punta in alto insieme a lui. Conservando
    # invece il q5 delle pose di erogazione, l'acceleratore restava inclinato e
    # sporgeva 540 mm in una direzione qualsiasi, spazzando il lettino durante
    # la rotazione: era questo - e non il braccio - a bloccare ogni transito.
    # q4 e q6 si possono conservare, perche' con q5 = 0 non inclinano nulla.
    w1 = np.array([q_from[0], q2, q3, q_from[3], 0.0, q_from[5]])
    w2 = np.array([q_to[0], q2, q3, q_to[3], 0.0, q_to[5]])
    if not (within_limits(w1) and within_limits(w2)):
        return None
    for a, b in ((q_from, w1), (w1, w2), (w2, q_to)):
        if path_collides(a, b, base, obstacles):
            return None
    return [w1, w2]


# ---------------------------------------------------------------------------
#  Sequenza di trattamento
# ---------------------------------------------------------------------------
def _interp_couch(a, b, t):
    """Orientamento intermedio del lettino fra due pose."""
    return CouchPose(a.yaw_deg + (b.yaw_deg - a.yaw_deg) * t,
                     a.pitch_deg + (b.pitch_deg - a.pitch_deg) * t,
                     a.roll_deg + (b.roll_deg - a.roll_deg) * t,
                     a.shift + (b.shift - a.shift) * t, a.pivot)


def escape_pool(base, couch_poses, obstacles, n_q1=13, n_q23=9):
    """Configurazioni candidate in cui il braccio puo' farsi da parte.

    Non una posa sola: durante la rotazione del lettino la zona libera si
    sposta, quindi serve un assortimento fra cui scegliere passo per passo.

    Si esplora una griglia grossolana sui primi tre giunti - quelli che
    decidono dove sta il braccio, con il polso dritto - e si tiene cio' che e'
    libero per almeno UNO degli orientamenti del lettino. Prima lo si ricavava
    da find_safe_shell per ogni orientamento: corretto ma inutilizzabile,
    perche' quella e' una scansione da decine di migliaia di valutazioni e qui
    va rifatta per ogni collocazione della base.
    """
    lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    g1 = np.linspace(lo[0], hi[0], n_q1)
    g2 = np.linspace(lo[1], hi[1], n_q23)
    g3 = np.linspace(lo[2], hi[2], n_q23)
    A, B, C = np.meshgrid(g1, g2, g3, indexing="ij")
    Q = np.zeros((A.size, 6))
    Q[:, 0], Q[:, 1], Q[:, 2] = A.ravel(), B.ravel(), C.ravel()
    keep = np.zeros(len(Q), dtype=bool)
    for couch in couch_poses:
        keep |= ~collides_batch(Q, couch.base_in_patient(base), obstacles)
    return Q[keep]


def couch_pose_attainable(base, obstacles, couch, pool, n_steps=8,
                         home=None):
    """Il lettino puo' ARRIVARE in quella posa senza travolgere il robot?

    Non basta che i fasci siano raggiungibili con il lettino gia' orientato:
    il lettino ci deve arrivare. Un piano da 2.2 metri che ruota di 90 gradi
    attorno all'isocentro spazza mezza sala all'altezza del braccio, e li' c'e'
    il robot: misurato, per quell'orientamento non esiste alcuna manovra
    valida, con nessun assortimento di vie di fuga (888 candidati, 20 passi).

    Il controllo chiede che a OGNI passo della rotazione il braccio abbia
    almeno un posto libero dove stare, e che i posti scelti in due passi
    consecutivi siano collegabili. Senza questo filtro l'ottimizzatore sceglie
    orientamenti magnifici sulla carta e inarrivabili nei fatti.
    """
    if len(pool) == 0:
        return False
    start = home if home is not None else CouchPose(pivot_mm=couch.pivot)
    ts = np.linspace(0.0, 1.0, n_steps + 1)
    chosen, q_prev = [], None
    for t in ts:
        c = _interp_couch(start, couch, t)
        ok = ~collides_batch(pool, c.base_in_patient(base), obstacles)
        if not ok.any():
            return False
        idx = np.nonzero(ok)[0]
        if q_prev is None:
            k = int(idx[0])
        else:
            k = int(idx[np.argmin(np.abs(pool[idx] - q_prev).sum(axis=1))])
        chosen.append(pool[k])
        q_prev = pool[k]
    for k in range(len(chosen) - 1):
        ca = _interp_couch(start, couch, ts[k])
        cb = _interp_couch(start, couch, ts[k + 1])
        if transfer_collides(chosen[k], chosen[k + 1], base, obstacles, ca, cb):
            return False
    return True


def couch_transition(q_from, c_from, q_to, c_to, base, obstacles, pool,
                     n_steps=6, n_try=8, max_checks=600):
    """Rotazione del lettino con il braccio che si sposta di CONTINUO.

    La manovra "parcheggia il braccio, poi ruota il lettino" non e' sempre
    possibile: misurato su questo piano, NON esiste alcuna configurazione libera
    contemporaneamente per tutti gli orientamenti attraversati. Nemmeno muovere
    braccio e lettino in linea retta funziona - urta in tutte e tre le
    transizioni.

    Quello che funziona e' cio' che farebbe il sistema reale: si suddivide la
    rotazione in passi, e a ogni passo il braccio si sposta nella zona che in
    quel momento e' libera. Ne esce un movimento coordinato e continuo, e ogni
    tratto viene verificato con il lettino che interpola al suo interno.

    La scelta del candidato a ogni passo e' una RICERCA, non una catena golosa:
    prima si prendeva sempre il libero piu' vicino al precedente e, se un solo
    tratto urtava, si rinunciava alla manovra anche con centinaia di
    alternative libere (650-950 per passo, misurate su lung_087). Ora si
    provano fino a `n_try` candidati per passo - i piu' vicini alla posa
    precedente e a quella d'arrivo - tornando indietro quando un ramo non porta
    a destinazione. `max_checks` limita i tratti verificati, perche' una
    manovra davvero impossibile non costi minuti. Quando e' la posa d'arrivo a
    non essere raggiungibile da nessuna via di fuga, la ricerca non basta: se
    ne occupa _couch_maneuver, passando dal parcheggio.

    Restituisce i punti di passaggio del braccio, oppure None.
    """
    if len(pool) == 0:
        return None
    ts = np.linspace(0.0, 1.0, n_steps + 1)
    couches = [_interp_couch(c_from, c_to, t) for t in ts]
    q_from = np.asarray(q_from, dtype=np.float64)
    q_to = np.asarray(q_to, dtype=np.float64)

    # Per ogni passo intermedio, quali candidati sono liberi.
    free = []
    for c in couches[1:-1]:
        ok = ~collides_batch(pool, c.base_in_patient(base), obstacles)
        if not ok.any():
            return None
        free.append(pool[ok])

    checks = [0]
    dead = set()            # (passo, candidato) da cui non si arriva

    def leg_free(a, b, k):
        # Tratto fra il passo k e il k+1, col lettino che ruota al suo interno.
        checks[0] += 1
        return not transfer_collides(a, b, base, obstacles,
                                     couches[k], couches[k + 1])

    def search(k, q_prev):
        # q_prev sta al passo k; manca la strada fino a q_to (passo n_steps).
        if k == n_steps - 1:
            return [] if leg_free(q_prev, q_to, k) else None
        cand = free[k]
        cost = (np.abs(cand - q_prev).sum(axis=1)
                + np.abs(cand - q_to).sum(axis=1))
        for j in np.argsort(cost)[:n_try]:
            j = int(j)
            if (k + 1, j) in dead:
                continue
            if checks[0] >= max_checks:
                return None
            if not leg_free(q_prev, cand[j], k):
                continue
            rest = search(k + 1, cand[j])
            if rest is not None:
                return [cand[j]] + rest
            dead.add((k + 1, j))
        return None

    return search(0, q_from)


# ---------------------------------------------------------------------------
#  Parcheggio: la strada di riserva fra due pose qualsiasi
# ---------------------------------------------------------------------------
def parking_poses(base, obstacles):
    """Vie di fuga libere per UNA configurazione di sala (lettino fermo)."""
    pool = escape_pool(base, [CouchPose()], obstacles)
    return pool[np.array([within_limits(q) for q in pool], dtype=bool)] \
        if len(pool) else pool


def parking_routes(q, base, obstacles, parks, n_try=12, n_keep=3):
    """Strade rettilinee da una posa di erogazione a una posa di parcheggio.

    Restituisce fino a `n_keep` strade, ciascuna come elenco di punti di
    passaggio che termina sul parcheggio (la posa `q` esclusa). Il primo tratto
    puo' passare da una variante "a polso allineato" del parcheggio - A1, A4 e
    A6 della posa, A5 a zero - che e' il ritiro di transit_route: la retta
    diretta verso un parcheggio a polso nullo fa ruotare la testata inclinata
    dentro il paziente.

    E' la definizione di posa COLLEGABILE: se esiste almeno una strada, dal
    parcheggio il braccio puo' arrivarci e ripartirne. Misurato su lung_087 a
    yaw 79: la prima posa del gruppo non si raggiungeva in linea retta da
    nessuna delle 200 vie di fuga provate, mentre con il ritiro allineato si'.
    """
    if not len(parks):
        return []
    q = np.asarray(q, dtype=np.float64)
    al = np.array(parks, dtype=np.float64)
    al[:, 0], al[:, 3], al[:, 4], al[:, 5] = q[0], q[3], 0.0, q[5]
    tip = np.array(parks, dtype=np.float64)
    tip[:, 3], tip[:, 4], tip[:, 5] = q[3], 0.0, q[5]
    routes = []
    # Prima i parcheggi raggiungibili direttamente, poi le varianti allineate:
    # a parita' di esito la strada con meno tratti e' la piu' breve.
    for kind, cand in (("diretta", parks), ("allineata", al),
                       ("allineata", tip)):
        ok = np.array([within_limits(c) for c in cand], dtype=bool)
        ok[ok] = ~collides_batch(cand[ok], base, obstacles)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            continue
        order = idx[np.argsort(np.abs(cand[idx] - q).sum(axis=1))][:n_try]
        for j in order:
            w = cand[j]
            if path_collides(q, w, base, obstacles):
                continue
            if kind == "diretta":
                routes.append([parks[j]])
            elif not path_collides(w, parks[j], base, obstacles):
                routes.append([w, parks[j]])
            if len(routes) >= n_keep:
                return routes
    return routes


def ik_solutions(source_world, beam_dir_world, base, n_roll=24, q_ref=None,
                 obstacles=None):
    """TUTTE le configurazioni valide per un fascio, dalla piu' comoda.

    solve_ik ne restituisce una sola; quando quella risulta intrappolata fra
    gli ostacoli conviene sapere se ce n'e' un'altra da cui si entra e si esce.
    """
    p_s = base.to_robot(source_world)
    n = base.dir_to_robot(beam_dir_world)
    n = n / max(np.linalg.norm(n), 1e-12)
    wrist = p_s - WRIST_TO_SOURCE * n
    out = []
    for q123 in _ik_position(wrist):
        for roll in np.linspace(0.0, 2 * np.pi, n_roll, endpoint=False):
            for q456 in _ik_orientation(q123, frame_from_axis(n, roll)):
                q = _resolve_turns(np.concatenate([q123, q456]), q_ref,
                                   MARGIN_FLOOR, MARGIN_BARRIER)
                if q is None or not within_limits(q):
                    continue
                if obstacles and clearance_mm(q, base, obstacles) < 0.0:
                    continue
                out.append(q)
    out.sort(key=lambda q: -limit_margin(q))
    return out


def connectable_pose(source_world, beam_dir_world, base, obstacles, parks,
                     n_roll=4, q=None):
    """Una posa per il fascio che sia anche COLLEGABILE al parcheggio.

    Restituisce (q, strade) oppure (None, []). Si prova prima `q` (o la
    soluzione di solve_ik), poi le altre soluzioni della IK. Lo usano sia il
    filtro della dosimetria sia l'assegnazione dei fasci agli angoli in
    solve_plan: un fascio ammesso dal planner deve essere uno che le manovre
    del lettino, che passano dal parcheggio, riescono a raggiungere.
    """
    if q is None:
        q, _ = solve_ik(source_world, beam_dir_world, base, n_roll=n_roll,
                        obstacles=obstacles)
    if q is None:
        return None, []
    if not obstacles:
        return q, []
    r = parking_routes(q, base, obstacles, parks)
    if r:
        return q, r
    for alt in ik_solutions(source_world, beam_dir_world, base,
                            n_roll=n_roll, obstacles=obstacles):
        if np.allclose(alt, q):
            continue
        r = parking_routes(alt, base, obstacles, parks)
        if r:
            return alt, r
    return None, []


def _sequence(beams, indices, base, n_roll, order_by_motion, margin_floor,
              margin_barrier, obstacles, check_path, q_start=None):
    """Ordina ed esegue un gruppo di fasci a lettino FERMO.

    E' il cuore che stava dentro solve_plan: qui la configurazione di sala e'
    una sola, quindi `base` e' gia' quella effettiva per l'angolo di lettino
    corrente. Restituisce (pose, statistiche, ultima_configurazione).
    """
    poses, q_prev = [], q_start
    blocked_pose, blocked_path, n_via = 0, 0, 0
    remaining = list(indices)

    shell = None
    if obstacles and check_path:
        shell = [c[:2] for c in find_safe_shell(base, obstacles)] or None

    while remaining:
        best_i, best_q, best_cost, best_via = None, None, np.inf, None
        pose_ko = path_ko = 0
        for i in remaining:
            b = beams[i]
            src = np.asarray(b.source, dtype=np.float64)
            direction = np.asarray(b.target, dtype=np.float64) - src
            q, info = solve_ik(src, direction, base, n_roll=n_roll,
                               q_ref=q_prev, margin_floor=margin_floor,
                               margin_barrier=margin_barrier,
                               obstacles=obstacles)
            if q is None:
                if info["scartate_per_collisione"] and info["entro_i_limiti"]:
                    pose_ko += 1        # cinematicamente ok, ma va a sbattere
                continue
            # Il TRASFERIMENTO dalla posa precedente deve essere libero: due
            # pose entrambe valide possono essere separate da una strada che
            # passa dentro il lettino.
            via = None
            if (check_path and obstacles and q_prev is not None and
                    path_collides(q_prev, q, base, obstacles)):
                # Strada diretta ostruita: si tenta il passaggio dalla posa di
                # ritiro. E' cio' che farebbe un operatore, e senza questa via
                # d'uscita il greedy si incastra e abbandona tutti i fasci
                # rimanenti anche se sono singolarmente eseguibili.
                via = transit_route(q_prev, q, shell, base, obstacles)
                if via is None:
                    path_ko += 1
                    continue
            if q_prev is None:
                cost = 0.0
            else:
                cost = float(np.sum(np.abs(q - q_prev)))
                if via is not None:
                    pts = [q_prev] + list(via) + [q]
                    cost = sum(float(np.sum(np.abs(v - u)))
                               for u, v in zip(pts[:-1], pts[1:]))
            if not order_by_motion:
                best_i, best_q, best_via = i, q, via
                break
            # A parita' di tutto si preferisce la strada diretta: il ritiro
            # costa tempo reale di trattamento.
            if cost < best_cost:
                best_i, best_q, best_cost, best_via = i, q, cost, via

        if best_i is None:
            # Nessuno dei rimanenti e' eseguibile da qui: si annota il MOTIVO,
            # perche' "irraggiungibile per geometria" e "bloccato da un
            # ostacolo" richiedono rimedi diversi.
            blocked_pose += pose_ko
            blocked_path += path_ko
            break

        remaining.remove(best_i)
        if best_via is not None:
            n_via += 1
        poses.append({"beam_index": best_i, "q": best_q,
                      "beam": beams[best_i],
                      "via": best_via,
                      "margine": limit_margin(best_q),
                      "franco_mm": (clearance_mm(best_q, base, obstacles)
                                    if obstacles else None)})
        q_prev = best_q

    stats = {"bloccati_da_ostacoli": blocked_pose,
             "bloccati_nel_percorso": blocked_path,
             "trasferimenti_con_ritiro": n_via,
             "non_eseguiti": list(remaining)}
    return poses, stats, q_prev


def _couch_maneuver(q_from, j_from, q_to, j_to, base, couch_poses, bases,
                    obstacles, pool, parks):
    """Manovra fra due gruppi: (punti di passaggio, tratti col lettino) o None.

    Prima si tenta la rotazione coordinata diretta fra le due pose di
    erogazione. Se fallisce, il braccio si ritira in parcheggio a lettino
    fermo, il lettino ruota con il braccio che si sposta fra vie di fuga, e la
    discesa sulla prima posa avviene a lettino gia' orientato. Misurato su
    lung_087: senza questa strada l'intero gruppo a yaw 79 (37 fasci) restava
    fuori, perche' la sua prima posa non si raggiunge da nessuna via di fuga
    con il lettino ancora in movimento; con questa, 108 fasci su 108. Serve
    anche per i residui: a yaw 79 lo spazio libero del braccio e' diviso in
    due dal lettino (A1 attorno a -106 e a +118 gradi), e l'unico modo di
    passare da una parte all'altra e' ruotare il lettino.

    I tratti col lettino in movimento sono restituiti come intervallo
    [inizio, fine) sugli indici dei tratti di [q_from] + via + [q_to]: gli
    altri avvengono a lettino fermo, all'angolo di partenza prima e a quello
    d'arrivo dopo. Senza questa informazione chi rilegge la manovra non puo'
    ricostruire dove stava il paziente in ciascun tratto.
    """
    if not obstacles:
        # Nulla da evitare: rotazione e trasferimento in un solo tratto.
        # Prima, con la pool vuota, la manovra risultava impossibile e il
        # gruppo veniva scartato senza motivo.
        return [], (0, 1)
    way = couch_transition(q_from, couch_poses[j_from], q_to,
                           couch_poses[j_to], base, obstacles, pool)
    if way is not None:
        return list(way), (0, len(way) + 1)
    if parks is None:
        return None
    ra = parking_routes(q_from, bases[j_from], obstacles, parks[j_from])
    rb = parking_routes(q_to, bases[j_to], obstacles, parks[j_to])
    for a in ra:
        for b in rb:
            way = couch_transition(a[-1], couch_poses[j_from], b[-1],
                                   couch_poses[j_to], base, obstacles, pool,
                                   max_checks=200)
            if way is not None:
                pts = list(a) + list(way) + list(reversed(b))
                return pts, (len(a), len(a) + len(way) + 1)
    return None


def solve_plan(beams, base, n_roll=24, order_by_motion=True,
               margin_floor=MARGIN_FLOOR, margin_barrier=MARGIN_BARRIER,
               obstacles=None, check_path=True, couch_poses=None):
    """Risolve la cinematica per tutti i fasci di un piano.

    Se `order_by_motion` e' attivo i fasci vengono riordinati con una euristica
    del nodo piu' vicino: il tempo di trattamento reale e' dominato dagli
    spostamenti del braccio fra un nodo e il successivo, non dall'erogazione.

    Con `couch_poses` il lettino diventa un ASSE MOBILE e puo' assumere piu'
    orientamenti durante la seduta. I fasci vengono allora RAGGRUPPATI per
    angolo di lettino - si ruota il paziente il minimo numero di volte - e fra
    un gruppo e l'altro il braccio si ritira in una posa di parcheggio sicura
    per tutti gli orientamenti attraversati.

    Restituisce (elenco_pose, diagnostica). Ogni posa porta l'indice
    dell'angolo di lettino da cui viene erogata.
    """
    n_simult = n_parked = 0
    if not couch_poses or len(couch_poses) <= 1:
        couch = couch_poses[0] if couch_poses else None
        b_eff = couch.base_in_patient(base) if couch is not None else base
        poses, stats, _ = _sequence(beams, range(len(beams)), b_eff, n_roll,
                                    order_by_motion, margin_floor,
                                    margin_barrier, obstacles, check_path)
        for p in poses:
            p["couch"] = 0
        groups = [len(poses)]
        n_unreach = len(beams) - len(poses)
    else:
        bases = [c.base_in_patient(base) for c in couch_poses]
        # Ogni fascio a UN angolo di lettino. Fra quelli che lo eseguono si
        # sceglie il gruppo piu' numeroso: meno gruppi significa meno
        # riposizionamenti del paziente.
        # Vie di fuga a lettino fermo, una serie per angolo. Entrano anche
        # nella matrice: un fascio vale per un angolo solo se da li' la sua
        # posa e' COLLEGABILE, altrimenti lo si assegnerebbe a un gruppo da
        # cui il braccio non riesce a raggiungerlo.
        parks = ([parking_poses(b, obstacles) for b in bases]
                 if obstacles and check_path else None)
        M, G = feasibility_matrix(beams, base, obstacles, couch_poses,
                                  n_roll=min(n_roll, 4), with_margin=True,
                                  parks=parks)
        order = [int(k) for k in np.argsort(-M.sum(axis=0))]
        assign = {j: [] for j in range(len(couch_poses))}
        for i in range(len(beams)):
            ok = [j for j in order if M[i, j]]
            if not ok:
                continue
            # Fra gli orientamenti che eseguono il fascio si prende quello che
            # lascia il braccio piu' lontano dai fine corsa. Prima si prendeva
            # il primo in ordine di produttivita': misurato, il polso A5
            # finiva esattamente sul limite e il margine mediano scendeva a
            # 0.087. Il numero di gruppi non cambia, perche' la scelta avviene
            # fra gli angoli gia' selezionati.
            assign[max(ok, key=lambda j: G[i, j])].append(i)

        # Assortimento di configurazioni in cui il braccio puo' farsi da parte
        # durante la rotazione del lettino. NON una posa sola: non ne esiste
        # una libera per tutti gli orientamenti insieme.
        pool = (escape_pool(base, couch_poses, obstacles)
                if obstacles and check_path else np.zeros((0, 6)))

        poses, groups, n_unreach = [], [], 0
        stats = {"bloccati_da_ostacoli": 0, "bloccati_nel_percorso": 0,
                 "trasferimenti_con_ritiro": 0}
        q_prev = None
        q_prev_group = last_couch = None
        # Un fascio bloccato nel percorso a un certo angolo di lettino puo'
        # essere comodissimo da un altro: i residui del primo giro vengono
        # ritentati altrove, finche' se ne recuperano. Senza questo secondo
        # giro restavano fuori 5 fasci su 100 che erano invece eseguibili.
        leftovers = []
        n_simult = 0
        rounds = list([k for k in order if assign[k]])
        for j in rounds:
            shell_j = ([c[:2] for c in find_safe_shell(bases[j], obstacles)]
                       if obstacles and check_path else None)
            if not poses:
                # Primo gruppo: si parte liberi, il braccio e' gia' fermo.
                gp, gs, q_prev = _sequence(beams, assign[j], bases[j], n_roll,
                                           order_by_motion, margin_floor,
                                           margin_barrier, obstacles,
                                           check_path)
            else:
                # Gruppi successivi: il lettino deve ruotare. Si calcola la
                # sequenza del gruppo e poi la manovra coordinata che ci porta
                # dentro; se non si trova, il gruppo resta non eseguito - meglio
                # dichiararlo che simulare un passaggio mai verificato.
                seq, st, qq = _sequence(beams, assign[j], bases[j], n_roll,
                                        order_by_motion, margin_floor,
                                        margin_barrier, obstacles, check_path)
                gp, gs, q_prev = [], st, qq
                if seq and q_prev_group is not None:
                    man = _couch_maneuver(q_prev_group, last_couch,
                                          seq[0]["q"], j, base, couch_poses,
                                          bases, obstacles, pool, parks)
                    if man is None:
                        st = dict(st or {}, non_eseguiti=list(assign[j]))
                        gs, gp, q_prev = st, [], None
                    else:
                        seq[0]["via"], seq[0]["couch_legs"] = man
                        seq[0]["couch_move"] = True
                        gp, q_prev = seq, qq
                        n_simult += 1
                        n_parked += int(man[1][0] > 0)
                elif seq:
                    gp, q_prev = seq, qq
            for p in gp:
                p["couch"] = j
            if gp:
                q_prev_group, last_couch = gp[-1]["q"], j
            poses.extend(gp)
            groups.append(len(gp))
            leftovers.extend(gs["non_eseguiti"])
            for k in ("bloccati_da_ostacoli", "bloccati_nel_percorso",
                      "trasferimenti_con_ritiro"):
                stats[k] += gs[k]

        # Secondo giro sui residui, agli altri angoli.
        for j in rounds:
            retry = [i for i in leftovers if M[i, j] and i not in
                     {p["beam_index"] for p in poses}]
            if not retry:
                continue
            shell_j = ([c[:2] for c in find_safe_shell(bases[j], obstacles)]
                       if obstacles and check_path else None)
            seq, st, qq = _sequence(beams, retry, bases[j], n_roll,
                                    order_by_motion, margin_floor,
                                    margin_barrier, obstacles, check_path)
            if not seq:
                continue
            if q_prev_group is None:
                continue
            man = _couch_maneuver(q_prev_group, last_couch, seq[0]["q"], j,
                                  base, couch_poses, bases, obstacles, pool,
                                  parks)
            if man is None:
                continue
            seq[0]["via"], seq[0]["couch_legs"] = man
            seq[0]["couch_move"] = True
            n_simult += 1
            n_parked += int(man[1][0] > 0)
            gp = seq
            for pz in gp:
                pz["couch"] = j
            q_prev_group, last_couch = gp[-1]["q"], j
            poses.extend(gp)
            groups.append(len(gp))
            q_prev = qq
        done = {p["beam_index"] for p in poses}
        n_unreach = len([i for i in range(len(beams)) if i not in done])

    # Movimento totale: e' il costo che l'ordinamento cerca di ridurre.
    total = 0.0
    for a, b in zip(poses[:-1], poses[1:]):
        pts = [a["q"]] + list(b.get("via") or []) + [b["q"]]
        total += sum(float(np.sum(np.abs(v - u)))
                     for u, v in zip(pts[:-1], pts[1:]))

    used = [g for g in groups if g]
    # Angoli DISTINTI, non gruppi: un angolo puo' essere ripreso nel secondo
    # giro, e contarlo due volte darebbe piu' riposizionamenti di quelli veri.
    n_angles = len({p["couch"] for p in poses}) if poses else 0
    n_moves = sum(1 for a, b in zip(poses[:-1], poses[1:])
                  if a["couch"] != b["couch"])
    diag = {
        "fasci_totali": len(beams),
        "raggiungibili": len(poses),
        "non_raggiungibili": n_unreach,
        "bloccati_da_ostacoli": stats["bloccati_da_ostacoli"],
        "bloccati_nel_percorso": stats["bloccati_nel_percorso"],
        "trasferimenti_con_ritiro": stats["trasferimenti_con_ritiro"],
        "manovre_dal_parcheggio": n_parked,
        "angoli_lettino": n_angles,
        "riposizionamenti_lettino": n_moves,
        "movimenti_lettino_simultanei": n_simult,
        "fasci_per_gruppo": used,
        "movimento_totale_deg": round(np.rad2deg(total), 1),
        "margine_minimo": round(min((p["margine"] for p in poses), default=0.0), 3),
    }
    if obstacles:
        fr = [p["franco_mm"] for p in poses if p["franco_mm"] is not None]
        diag["franco_minimo_mm"] = round(min(fr), 1) if fr else None
    return poses, diag


def deliverable_subset(beams, base, obstacles, couch_poses=None, n_roll=4,
                       max_rounds=3, status_cb=None):
    """I fasci che la sequenza del robot esegue DAVVERO: (indici, diagnostica).

    E' lo stesso calcolo del nodo ROS (solve_plan, n_roll=4). Si ripete sul
    sottoinsieme finche' tutti i fasci rimasti vengono erogati: togliere un
    fascio cambia l'ordine della sequenza, e un solo passaggio non garantisce
    che il sottoinsieme sia a sua volta eseguibile per intero.

    Serve alla dosimetria per calcolare la dose finale solo su questi fasci.
    Misurato su lung_085 da riga di comando: senza questo filtro il nodo
    erogava 114 fasci su 116 e la copertura scendeva dal 94.9% pianificato
    al 57.8%; con il filtro 114 su 114 e 94.9% in entrambi i casi.
    """
    idx = list(range(len(beams)))
    diag = {}
    for k in range(max_rounds):
        poses, diag = solve_plan([beams[i] for i in idx], base, n_roll=n_roll,
                                 obstacles=obstacles, couch_poses=couch_poses)
        ok = sorted(p["beam_index"] for p in poses)
        if status_cb:
            status_cb(f"Collaudo di erogabilita' {k + 1}: {len(ok)}/{len(idx)} "
                      f"fasci eseguiti dalla sequenza del robot")
        if len(ok) == len(idx):
            diag["verificato"] = True
            return idx, diag
        idx = [idx[i] for i in ok]
    # Ultimo sottoinsieme ristretto ma non piu' ricontrollato: lo si dice,
    # invece di spacciarlo per eseguibile.
    diag["verificato"] = False
    return idx, diag


def reachable_count(beams, base, obstacles, n_roll=4, couch=None,
                    stop_below=None):
    """Quanti fasci sono eseguibili, e con che franco. Serve da funzione
    obiettivo alla ricerca della configurazione di sala.

    Si valutano i fasci uno per uno, senza ordinamento ne' controllo dei
    percorsi: qui interessa solo se la posa esiste. Il percorso si verifica
    dopo, su una sola configurazione, perche' costa molto di piu'.

    `stop_below` interrompe appena e' matematicamente impossibile superare
    quella soglia: con qualche centinaio di combinazioni da provare, abbandonare
    presto le peggiori vale piu' di qualunque micro-ottimizzazione.
    """
    b_eff = couch.base_in_patient(base) if couch is not None else base
    ok, fr = 0, []
    n = len(beams)
    for i, b in enumerate(beams):
        q, _ = solve_ik(b.source, b.target - b.source, b_eff,
                        n_roll=n_roll, obstacles=obstacles)
        if q is not None:
            ok += 1
            fr.append(clearance_mm(q, b_eff, obstacles))
        elif stop_below is not None and ok + (n - i - 1) < stop_below:
            return ok, 0.0
    return ok, (min(fr) if fr else 0.0)


def sample_beam_directions(pivot_mm, n=400, couch_block_deg=40.0, sad=800.0):
    """Direzioni di fascio ben distribuite attorno al bersaglio.

    Riproduce i nodi che genera la Fase 2 (spirale di Fibonacci, cono
    posteriore del lettino escluso). Serve a scegliere gli angoli di lettino
    senza dover prima costruire tutti i candidati veri, che sono migliaia.
    """
    pivot = np.asarray(pivot_mm, dtype=np.float64)
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = np.pi * (1.0 + 5.0 ** 0.5) * i
    u = np.stack([np.cos(th) * np.sin(phi), np.cos(phi),
                  np.sin(th) * np.sin(phi)], axis=1)
    u = u[u[:, 1] >= np.cos(np.deg2rad(180.0 - couch_block_deg))]
    return [_Ray(pivot + sad * v, pivot.copy()) for v in u]


class _Ray:
    """Fascio minimo: solo sorgente e bersaglio, quanto basta alla IK."""

    __slots__ = ("source", "target")

    def __init__(self, source, target):
        self.source, self.target = source, target


def feasibility_matrix(beams, base, obstacles, couch_poses, n_roll=4,
                       status_cb=None, with_margin=False, parks=None):
    """Matrice (fasci x angoli di lettino): chi e' eseguibile e da dove.

    Con `with_margin` restituisce anche, per ogni coppia, il margine dai fine
    corsa della configurazione trovata. Serve all'assegnazione: lo stesso
    fascio eseguito da due orientamenti diversi del lettino puo' lasciare il
    braccio comodo o schiacciato sul fine corsa, e scegliere solo per copertura
    lo lascia schiacciato.

    Con `parks` (vie di fuga per angolo, da parking_poses) conta solo una posa
    COLLEGABILE: e' il criterio della sequenza, e la matrice deve usare lo
    stesso, altrimenti un fascio finisce nel gruppo sbagliato.
    """
    M = np.zeros((len(beams), len(couch_poses)), dtype=bool)
    G = np.full((len(beams), len(couch_poses)), -1.0)
    for j, couch in enumerate(couch_poses):
        if status_cb:
            status_cb(f"Angolo di lettino {j + 1}/{len(couch_poses)}...")
        b_eff = couch.base_in_patient(base)
        for i, b in enumerate(beams):
            if parks is not None:
                q, _ = connectable_pose(b.source, b.target - b.source, b_eff,
                                        obstacles, parks[j], n_roll=n_roll)
            else:
                q, _ = solve_ik(b.source, b.target - b.source, b_eff,
                                n_roll=n_roll, obstacles=obstacles)
            if q is not None:
                M[i, j] = True
                G[i, j] = limit_margin(q)
    return (M, G) if with_margin else M


def _greedy_cover(M, max_poses, tie_cost):
    """Pochi angoli che coprono piu' fasci possibile. Insieme di copertura.

    Il problema e' NP-difficile, ma la scelta golosa (a ogni passo l'angolo che
    aggiunge piu' fasci nuovi) e' quella classica e qui basta: gli angoli
    candidati sono poche decine e le differenze fra soluzioni vicine sono
    minime rispetto al costo di ruotare il paziente.
    """
    chosen, covered = [], np.zeros(M.shape[0], dtype=bool)
    while len(chosen) < max_poses:
        gain = (M & ~covered[:, None]).sum(axis=0)
        gain[chosen] = -1
        best = int(np.lexsort((tie_cost, -gain))[0])
        if gain[best] <= 0:
            break
        chosen.append(best)
        covered |= M[:, best]
    return chosen, covered


def optimize_setup(obstacles, pivot_mm, base_template=None,
                   x_values=None, z_values=None,
                   yaw_values=None, pitch_values=None,
                   max_couch_poses=3, n_roll=4, probe=None,
                   couch_arm=None, status_cb=None):
    """Cerca INSIEME la collocazione della base e gli angoli del lettino.

    Non sono scelte separabili: la posizione migliore per la base dipende da
    come e' orientato il paziente, e viceversa.

    UN SOLO ANGOLO DI LETTINO NON BASTA - misurato su 353 direzioni di fascio
    ben distribuite, con la base ottima: lettino fermo 25% eseguibili, miglior
    angolo singolo 27%. Il guadagno vero arriva facendo assumere al lettino
    POCHI ANGOLI DISTINTI durante il trattamento: tre bastano per arrivare al
    45%, cioe' quasi il doppio delle direzioni disponibili. E' anche il modo in
    cui si tratta davvero, raggruppando i fasci per angolo di lettino.

    Si cerca solo la ROTAZIONE del piano (yaw). Il paziente non viene
    inclinato: `pitch_values` resta a zero per scelta, non per semplificazione
    (vedi la nota sopra le costanti del lettino per i numeri).

    La scelta si fa su un campione di direzioni (`probe`), non sui candidati
    veri della Fase 2, che sono migliaia: gli angoli buoni dipendono dalla
    geometria della sala, non da quali fasci l'ottimizzatore preferira'.

    Restituisce base, elenco di angoli di lettino e diagnostica.
    """
    base_template = base_template or RobotBase()
    fy = float(base_template.origin[1])
    x_values = [-1000.0, -900.0, -800.0, -700.0] if x_values is None else x_values
    z_values = [-450.0, -300.0, -150.0, 0.0] if z_values is None else z_values
    if yaw_values is None:
        # Griglia fitta: lo yaw e' l'unica variabile di orientamento che paga,
        # quindi tanto vale darle risoluzione. 13 valori da -95 a +95 gradi.
        yaw_values = list(np.linspace(-COUCH_YAW_RANGE_DEG,
                                      COUCH_YAW_RANGE_DEG, 13))
    if pitch_values is None:
        # Nessuna INCLINAZIONE del paziente: il pitch resta un asse di
        # correzione del polso, non una variabile di ottimizzazione.
        pitch_values = [0.0]

    probe = probe if probe is not None else sample_beam_directions(pivot_mm)
    poses = [CouchPose(a, b, 0.0, (0.0, 0.0, 0.0), pivot_mm)
             for a in yaw_values for b in pitch_values]
    # Filtro di REALIZZABILITA': il lettino e' un manipolatore, non una
    # piattaforma che assume qualunque orientamento. Senza questo controllo
    # l'ottimizzatore proporrebbe pose che nessun meccanismo raggiunge.
    if couch_arm is not None:
        keep = [c for c in poses if couch_arm.solve_pose(c) is not None]
        if status_cb:
            status_cb(f"Orientamenti realizzabili dal lettino: "
                      f"{len(keep)}/{len(poses)}")
        poses = keep or [CouchPose(pivot_mm=pivot_mm)]
    n_mech = len(poses)
    # A parita' di guadagno si preferisce il lettino meno ruotato: girare il
    # paziente costa tempo, e l'angolo zero non richiede nemmeno di muoverlo.
    tie = np.array([abs(c.yaw_deg) + 2.0 * abs(c.pitch_deg) for c in poses])

    best = None
    for xi, x in enumerate(x_values):
        for z in z_values:
            base = RobotBase(origin_mm=(x, fy, z), yaw_deg=0.0)
            # Filtro di RAGGIUNGIBILITA' del lettino: gli orientamenti da cui
            # il lettino non riesce ad arrivare senza travolgere il robot vanno
            # scartati PRIMA di valutarli, non dopo.
            usable = poses
            if obstacles:
                pool = escape_pool(base, poses, obstacles)
                usable = [c for c in poses
                          if c.is_identity()
                          or couch_pose_attainable(base, obstacles, c, pool)]
                if not usable:
                    continue
            if status_cb:
                status_cb(f"Base X={x:.0f} Z={z:.0f}: "
                          f"{len(usable)}/{n_mech} orientamenti raggiungibili")
            M = feasibility_matrix(probe, base, obstacles, usable,
                                   n_roll=n_roll)
            tie_u = np.array([abs(c.yaw_deg) + 2.0 * abs(c.pitch_deg)
                              for c in usable])
            chosen, covered = _greedy_cover(M, max_couch_poses, tie_u)
            score = (int(covered.sum()), -len(chosen),
                     -float(sum(tie_u[k] for k in chosen)))
            if best is None or score > best[0]:
                best = (score, base, [usable[k] for k in chosen],
                        int(covered.sum()), len(usable))

    score, base, schedule, covered_n, n_usable = best
    # Confronto onesto: quanto si otterrebbe con il lettino fermo, da qui.
    fixed = CouchPose(pivot_mm=pivot_mm)
    n_fixed, _ = reachable_count(probe, base, obstacles, n_roll=n_roll,
                                 couch=fixed)
    return {
        "base": base,
        "couch_poses": schedule,
        "angoli_realizzabili": n_mech,
        "angoli_raggiungibili": n_usable,
        "copertura_direzioni": covered_n,
        "copertura_lettino_fermo": int(n_fixed),
        "direzioni_provate": len(probe),
        "angoli_valutati": len(poses),
    }


def fk_error_mm(q, source_world, beam_dir_world, base):
    """Errore di chiusura: rifa' la diretta e confronta con la posa richiesta.

    E' il controllo che una IK deve sempre superare: se la soluzione, ripassata
    per la cinematica diretta, non riproduce la posa di partenza, la soluzione
    e' sbagliata anche quando sembra plausibile.
    """
    src, d, _ = forward_kinematics(q, base)
    n_req = np.asarray(beam_dir_world, dtype=np.float64)
    n_req = n_req / max(np.linalg.norm(n_req), 1e-12)
    pos_err = float(np.linalg.norm(src - np.asarray(source_world, dtype=np.float64)))
    ang_err = float(np.rad2deg(np.arccos(np.clip(np.dot(d, n_req), -1.0, 1.0))))
    return pos_err, ang_err
