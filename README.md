# CyberKnife Control Center — radiochirurgia robotica del tumore polmonare, simulata

Il progetto simula da capo a fondo un trattamento di **radiochirurgia robotica**
ispirato al sistema CyberKnife: si parte da una TAC del torace e si arriva a un
braccio robotico che eroga il piano di dose su un tumore che respira.

```
  TAC del paziente
        │
        ▼
  1 PERCEZIONE ──────► 2 RESPIRO E TECNICA ──────► 3 DOSIMETRIA ──────► 4 CINEMATICA E ROS 2
  rete neurale 3D       moto del tumore,             piano di fasci,       il robot esegue il piano
  trova il tumore e     scelta della tecnica         dose nei tessuti,     in RViz e scrive il
  dice quanto è sicura  (ITV, gating, inseguimento)  organi a rischio      resoconto della seduta
```

Tutto passa da un'unica applicazione desktop, il **CyberKnife Control Center**
(PyQt6), con quattro schede che seguono l'ordine della pipeline. Ogni scheda
consegna il suo risultato a quella dopo: il contorno del tumore diventa il
bersaglio, la tecnica di gestione del moto decide il margine, il piano di dose
diventa una sequenza di pose del robot, e la simulazione ROS 2 restituisce
quanta dose è arrivata davvero.

Il progetto è didattico e di ricerca: le dosi sono in scala relativa, il
respiro è sintetico (il dataset contiene TAC statiche) e il robot è simulato.
Ogni approssimazione è dichiarata nel codice, accanto al punto in cui viene
fatta.

---

## Indice

1. [Che cosa fa, in sintesi](#che-cosa-fa-in-sintesi)
2. [Avvio](#avvio)
3. [Guida all'applicativo](#guida-allapplicativo)
   - [Finestra principale, menu e progetti](#finestra-principale-menu-e-progetti)
   - [Scheda 1 — Acquisizione e Percezione](#scheda-1--acquisizione-e-percezione)
   - [Scheda 2 — Respiro e Tecnica](#scheda-2--respiro-e-tecnica)
   - [Scheda 3 — Dosimetria e Simulazione Raggi](#scheda-3--dosimetria-e-simulazione-raggi)
   - [Scheda 4 — Cinematica e Controllo ROS 2](#scheda-4--cinematica-e-controllo-ros-2)
   - [La simulazione in RViz](#la-simulazione-in-rviz)
   - [Il resoconto di fine seduta](#il-resoconto-di-fine-seduta)
4. [Come funziona dentro](#come-funziona-dentro)
5. [Risultati principali](#risultati-principali)
6. [Pipeline da riga di comando](#pipeline-da-riga-di-comando)
7. [Test](#test)
8. [Installazione](#installazione)
9. [Dataset](#dataset)
10. [Struttura del progetto](#struttura-del-progetto)

---

## Che cosa fa, in sintesi

| Fase | Che cosa succede | Moduli principali |
|---|---|---|
| **Percezione** | Una UNet 3D (PyTorch + MONAI) segmenta il tumore nella TAC. L'analisi di incertezza ripete la predizione su 8 versioni perturbate della TAC; un **semaforo** dice se il bersaglio è affidabile | `inference_worker.py`, `uncertainty.py`, `postproc.py` |
| **Anatomia** | Polmoni, vie aeree, mediastino, ossa e profilo corporeo vengono segmentati per densità (unità Hounsfield) e ricostruiti in 3D | `organs.py`, `organ_mesh.py` |
| **Respiro** | Il tumore si muove con un respiro sintetico realistico; si simula il sistema di inseguimento (Synchrony): sensori sul torace, radiografie periodiche, modello di correlazione, predittore della latenza | `motion_model.py`, `motion_predictor.py`, `motion_dose.py` |
| **Dosimetria** | Un pianificatore inverso a cinque stadi sceglie i fasci (posizione, collimatore, peso) e calcola la dose con un modello TERMA + convoluzione sulla densità della TAC | `planner.py`, `dosimetry.py` |
| **Robot** | Cinematica inversa in forma chiusa del manipolatore (KUKA KR 300 R2500 ultra) e del lettino robotico, collisioni, ottimizzazione della sala | `kinematics.py` |
| **Simulazione** | Il piano viene esportato in JSON a un nodo ROS 2 (in WSL) che anima il robot in RViz, insegue il tumore e scrive il resoconto | `plan_export.py`, `ros2_ws/` |

---

## Avvio

```bash
python main.py
```

(oppure `.venv\Scripts\python.exe main.py`). Serve solo una TAC in formato
NIfTI (`.nii.gz`) e il modello già presente in `outputs/model_final_all.pth`.
L'installazione dell'ambiente è descritta [più sotto](#installazione).

Un flusso completo tipico, su `lung_001` del dataset:

1. **Scheda 1** → *Carica Scansione CT* → scegli la TAC. Dopo ~25 s compaiono
   tumore, coordinate e semaforo.
2. **Scheda 2** → scegli la *Tecnica di gestione del moto* (per esempio
   *Inseguimento (Synchrony)*) e premi *Simula seduta* per vedere quanto bene
   il robot segue il tumore.
3. **Scheda 3** → *Genera Modello 3D* (~30–60 s), poi *Calcola Dosimetria*
   (~1–2 min, include l'ottimizzazione della sala).
4. **Scheda 4** → *Risolvi cinematica*, poi *Lancia simulazione ROS 2 / RViz*.
   A fine seduta si apre da solo il resoconto.
5. **File → Salva progetto** per riaprire tutto più tardi senza ricalcolare.

---

## Guida all'applicativo

### Finestra principale, menu e progetti

| Comando | Scorciatoia | Che cosa fa |
|---|---|---|
| File → **Carica scansione CT...** | Ctrl+O | Equivale al pulsante della scheda 1 |
| File → **Apri progetto...** | Ctrl+Maiusc+O | Riporta l'applicazione al punto in cui era: maschera, incertezza e semaforo, tecnica e margine, modello 3D, piano, cinematica, ultimo resoconto ROS |
| File → **Salva progetto** | Ctrl+S | Salva i risultati di tutte le fasi calcolate finora |
| File → **Salva progetto con nome...** | Ctrl+Maiusc+S | Come sopra, in un altro file |
| File → **Esci** | Ctrl+Q | Chiede conferma se ci sono risultati non salvati o calcoli in corso |
| ? → **Informazioni** | | Nome e versione dell'applicazione |

I tre comandi sui progetti sono anche nella barra in alto. Il salvataggio
viene proposto **a fine pipeline** (quando la seduta ROS si chiude con il
resoconto) e **alla chiusura**.

La **barra di stato** in basso dice quale motore disegna la vista 3D
(`3D: GPU (VTK)` oppure `3D: Matplotlib (ripiego)`, molto più lento) e se ROS 2
è disponibile in WSL. Se manca, la scheda 4 funziona lo stesso ma non può
aprire RViz.

**I progetti (`.ckproj`)** finiscono in `outputs/progetti/`. Sono archivi npz
compressi, tipicamente pochi MB, con un JSON che descrive lo stato. Non
contengono pickle, quindi aprire un progetto ricevuto da altri non esegue
codice. La TAC **non** viene copiata: si salva il percorso con dimensione e
data, e all'apertura il programma avvisa se il file è cambiato. Ciò che si
ricalcola in un secondo (griglia di densità, mappa di dose) non viene salvato.
Se qualcosa non si può salvare, il programma lo elenca invece di perderlo in
silenzio.

### Scheda 1 — Acquisizione e Percezione

Trova il tumore nella TAC e dice quanto fidarsi del risultato.

**Comandi**

| Controllo | Effetto |
|---|---|
| **Carica Scansione CT (.nii.gz)** | Avvia la segmentazione in background, con barra di avanzamento |
| **Analisi di incertezza** (attiva di default) | Ripete la predizione su 8 versioni perturbate della TAC (specchiamento, piccole rotazioni e scale, rumore: le stesse trasformazioni viste in addestramento). ~10 s in più su GPU. Alimenta semaforo, mappa di incertezza e PTV adattivo |
| **Finestra HU** | Preset di contrasto: Polmone, Mediastino, Osso, Cervello. Cambia solo la resa in grigi, non i dati |
| **Evidenzia tumore** | Sovrappone il contorno della lesione |
| **Mappa isodose** | Sovrappone la dose; si attiva dopo il calcolo nella scheda 3 |
| **Mappa incertezza** | Deviazione standard della probabilità fra le predizioni perturbate: chiaro = la rete cambia idea |
| **Zoom sul tumore** / **Vai al tumore** | Centra le tre viste sulla lesione |
| **Esporta Target (.txt)** | Scrive coordinate e dati della lesione in un file di testo |

Le **viste ortogonali** (assiale, coronale, sagittale) scorrono con la rotella
del mouse o con gli slider.

**Il pannello «Dati Target»** mostra:

- il **semaforo** del bersaglio, con i motivi:
  - 🟢 **affidabile**: la lesione supera il cancello sul volume e le
    predizioni perturbate sono concordi;
  - 🟡 **da rivedere**: in qualche predizione perturbata la rete sceglie
    un'altra lesione a più di 50 mm (esiste un candidato alternativo);
  - 🔴 **non affidabile**: lesione troppo piccola (sotto 1 cm³) o assente;
- le **coordinate reali** del centro del tumore in millimetri (sistema RAS del
  paziente: +X destra, +Y anteriore, +Z craniale) e in voxel;
- volume, voxel e numero di lesioni trovate (componenti grezze e lesioni oltre
  la soglia di volume);
- dispersione dell'isocentro e del volume fra le predizioni perturbate.

Un bersaglio non affidabile resta visibile e utilizzabile — il simulatore serve
anche a studiare i fallimenti — ma viene marcato in arancio.

### Scheda 2 — Respiro e Tecnica

Un tumore polmonare si sposta a ogni respiro, fino a ~18 mm vicino al
diaframma. Questa scheda sceglie **come gestire il moto**, e la scelta decide
il bersaglio della dosimetria e il comportamento del robot in ROS. Per questo
viene prima della dosimetria.

**Tecnica di gestione del moto**

| Tecnica | Bersaglio della dosimetria | Nella simulazione ROS |
|---|---|---|
| Nessuna: tumore considerato fermo | CTV (il contorno della rete) | tumore immobile; utile solo come riferimento |
| ITV: margine sull'intera traiettoria, fascio fermo | inviluppo del respiro (CTV + 5 mm lungo la traiettoria di una 4DCT simulata) | il tumore respira, il fascio no; il lettino corregge la deriva |
| Gating: fascio fermo, spara solo in finestra | CTV + margine dell'errore a fascio acceso | il robot aspetta e spara solo quando il tumore stimato è nella finestra |
| **Inseguimento (Synchrony)** | CTV + margine di van Herk dell'errore residuo | il robot insegue il tumore |
| Inseguimento + margine adattivo all'incertezza | CTV + corona dove la rete è incerta + margine combinato | il robot insegue il tumore |

**Respiro del paziente**

- **Ampiezza dalla posizione della lesione** (attiva di default): l'escursione
  si ricava da dove sta il tumore nel polmone; i polmoni vengono segmentati in
  background in pochi secondi.
- **Escursione SI**, **Periodo**, **Irregolarità** (variabilità da un ciclo
  all'altro), **Deriva** (spostamento lento della posizione di riposo).

**Sistema di inseguimento**

- **Predittore** della latenza di 115 ms: nessuno, lineare, Kalman, nLMS, GRU
  (rete neurale). Nel nodo ROS la GRU non è disponibile e si usa Kalman.
- **Radiografie ogni** *N* secondi: ogni quanto una coppia di immagini kV
  misura il tumore e ritara il modello di correlazione. È la leva principale
  sulla precisione (p95 3.2 mm a 60 s, 2.6 mm a 30 s).
- **Seme** del generatore casuale, per ripetere lo stesso respiro.

**Strumenti**

| Pulsante | Che cosa risponde | Tempo |
|---|---|---|
| **Simula seduta (5 min)** | Il robot riesce a seguire il tumore? Tracce del respiro, posizione stimata, errore d'inseguimento, confronto fra i predittori (sotto-scheda *Inseguimento*) | pochi secondi |
| **Confronta tecniche (dose)** | Che cosa cambia per la dose? Un piano completo per scenario (statico, CTV senza compensazione, ITV, inseguimento; con **Includi il PTV adattivo** anche quello), valutato sul tumore in movimento (sotto-scheda *Dose sotto moto*) | ~1 min per scenario |

### Scheda 3 — Dosimetria e Simulazione Raggi

Ricostruisce l'anatomia in 3D e calcola il piano di trattamento. Ha quattro
sotto-schede.

#### Modello 3D

| Controllo | Effetto |
|---|---|
| **Genera Modello 3D** | Segmenta e ricostruisce corpo, polmoni, vie aeree, mediastino, scheletro e tumore (~30–60 s) |
| **Esporta STL** | Salva le mesh |
| **Vista** | Isometrica, Anteriore, Laterale, Superiore |
| **Trasparenza busto**, caselle degli organi, **Tumore** | Mostrano o nascondono le singole strutture |
| Mouse | Trascina per ruotare, rotella per lo zoom, tasto centrale per spostare |

**Parametri del piano**, sopra la vista:

| Controllo | Effetto |
|---|---|
| **Piano** | *Conformale (coni variabili)*, il piano CyberKnife vero con molti punti di mira; oppure *Isocentro singolo (sferico)*, un solo punto di mira con un collimatore scelto a mano |
| **Bersaglio** | *Secondo la tecnica (Tab 2)* (default), oppure per confronto *CTV predetto*, *CTV + 5 mm*, *PTV adattivo, solo incertezza* |
| **Collimatore** | Diametro del cono, per il piano a isocentro singolo |
| **Nodi** | Posizioni del robot usate dal piano a isocentro singolo |
| **Budget fasci** | Numero massimo di fasci del piano conformale |
| **Calcola Dosimetria** | Ottimizza la sala (base del robot e angoli del lettino), sceglie solo fasci che il robot può eseguire e calcola la dose (~1–2 min) |
| **Mostra fasci**, **Superficie isodose**, **Livello** | Disegnano i fasci (spessore = peso) e la superficie a una data percentuale della dose massima |

Il **pannello laterale** riassume il piano: numero di fasci e unità monitor,
**conformità** (indice di conformità, gradiente), **copertura** del bersaglio
(V100, D95, dose minima), **percorso nei tessuti** (osso attraversato),
sezione **ROBOT** (base e angoli di lettino adottati, fasci eseguibili), gli
**stadi dell'ottimizzazione** e la **dose agli organi** a rischio.

#### Sezione (dove passano)

Un solo piano della TAC (assiale, coronale o sagittale) a piena risoluzione,
con sopra **contorni degli organi**, **fasci** con la **larghezza reale del
cono**, **isodosi** e **punti d'ingresso**. Un fascio che attraversa troppo
osso è segnalato in rosso. Il riquadro *Fasci in questa sezione* elenca i
fasci che giacciono davvero nel piano.

#### Vista dal fascio (BEV)

La scena vista dalla sorgente di un singolo fascio, come nella verifica
clinica fascio per fascio. Si scorre fra i fasci con ◀ ▶ o dal menu, e si
possono attivare **Radiografia (DRR)** (radiografia simulata), **Bersaglio**,
**Ossa**, **Apertura collimatore**. L'*Analisi del fascio* risponde a due
domande: il fascio inquadra tutto il bersaglio? E per arrivarci passa
dall'osso?

#### Istogramma Dose-Volume (DVH)

Curve dose-volume del tumore e degli organi (polmoni, vie aeree, mediastino,
ossa). Con l'analisi di incertezza attiva compare una **banda** attorno alla
curva del tumore: i DVH che si otterrebbero con contorni più stretti o più
larghi (soglie 0.3–0.7 sulla probabilità media).

### Scheda 4 — Cinematica e Controllo ROS 2

Trasforma il piano in una sequenza di pose eseguibili dal robot e lancia la
simulazione.

| Controllo | Effetto |
|---|---|
| **Collocazione del robot**: X, Y, Z, Rotazione | Posizione della base del manipolatore in sala (mm, sistema del paziente). Dopo *Calcola Dosimetria* viene adottata quella ottimizzata |
| **Rotazioni provate** | Quante rotazioni attorno all'asse del fascio provare: il collimatore è circolare, quindi danno lo stesso fascio con configurazioni di giunto diverse |
| **Angoli di lettino** | Quanti orientamenti può assumere il lettino robotico durante la seduta (1 = fermo). Più angoli rendono raggiungibili più direzioni |
| **Ottimizza sala (base + lettino)** | Cerca insieme posizione della base e orientamenti del lettino |
| **Risolvi cinematica** | Cinematica inversa per ogni fascio, ordine di erogazione, collisioni con lettino e paziente, percorsi fra una posa e l'altra |
| **Esporta piano per ROS (JSON)** | Salva il piano dove si preferisce (per la simulazione non serve) |
| **Lancia simulazione ROS 2 / RViz** | Esporta piano, scena e URDF nel pacchetto ROS, compila se serve e avvia la simulazione in WSL. Resta disattivato mentre una simulazione è in corso |
| **Apri resoconto dell'ultima seduta** | Riapre la finestra di resoconto |

I grafici mostrano, fascio per fascio nell'ordine di erogazione, gli **angoli
dei sei giunti** e il **margine dai fine corsa**. Il pannello riporta quanti
fasci sono eseguibili, il franco minimo dagli ostacoli e il movimento totale.
Se dopo la dosimetria si cambia la base o il lettino, un avviso ricorda che i
fasci erano stati scelti per l'altra sala.

### La simulazione in RViz

La simulazione gira in **WSL2 / Ubuntu 22.04 con ROS 2 Humble**. Il lancio
apre tre finestre:

- **la sala**: robot, lettino robotico, anatomia del paziente, tutti i fasci
  del piano;
- **Beam's Eye View**: la camera sta sulla sorgente e guarda lungo il fascio
  in erogazione;
- **il tumore da vicino**: ogni voxel parte rosso e diventa arancio, giallo e
  infine **verde quando ha ricevuto la dose prescritta**; i fasci lasciano
  tunnel colorati che si sommano dove si sovrappongono.

| Che cosa si vede | Significato |
|---|---|
| Fasci grigio-azzurri | tutti i fasci del piano; lo spessore è il peso |
| Fascio giallo con alone conico | il fascio in erogazione adesso, con il diametro reale del collimatore |
| Fasci verdi | già erogati |
| Lastra scura semitrasparente | lettino: volume proibito dal pavimento al piano |
| Riquadro verde a spigoli | volume di esclusione del paziente |

Se il piano contiene il respiro (tecnica diversa da «Nessuna»), il tumore si
muove e il robot lo insegue, aspetta (gating) o resta fermo (ITV) secondo la
tecnica scelta nella scheda 2. Il topic `cyberknife/inseguimento` pubblica a
ogni passo offset vero, offset del fascio, errore e sospensione. Erogato
l'ultimo fascio, la simulazione si chiude da sola.

Si può avviare anche a mano, da WSL:

```bash
bash ros2_ws/run.sh                                    # come dal pulsante
bash ros2_ws/run.sh rviz:=false bev:=false             # senza finestre
bash ros2_ws/run.sh dwell_s:=0.1 joint_speed_deg_s:=360.0   # animazione veloce
bash ros2_ws/run.sh loop:=true                         # ripete all'infinito
bash ros2_ws/run.sh inseguimento:=false                # il tumore respira, il fascio no
bash ros2_ws/run.sh respiro:=false                     # tumore immobile
```

Altri argomenti: `tumore` (finestra del tumore, default `true`),
`final_hold_s` (pausa finale, 4 s), `turntable_deg_s` (rotazione della camera
sul tumore, 12 °/s), `dose_view_hz` (aggiornamenti dei colori, 12 Hz),
`erogazione_s` (tempo totale di fascio acceso, ripartito fra i fasci in
proporzione al peso).

### Il resoconto di fine seduta

A fine simulazione il nodo ROS scrive `resoconto_trattamento.json` e la scheda
4 apre la finestra di resoconto:

- **Sintesi**: fasci erogati su pianificati, bersaglio alla dose prescritta
  contro il pianificato, D95, dose minima, durata (spostamenti, erogazione,
  rotazioni del lettino), angoli di lettino usati, franco e margini minimi;
  con il respiro attivo, errore d'inseguimento e dose persa.
- **Replay del tumore**: gli stessi voxel della finestra RViz che si
  riempiono fascio dopo fascio. *Riproduci*, *Inizio*, *Fine*, cursore,
  velocità da 0.5× a 8×.
- **Andamento del bersaglio**: copertura, D95, dose minima e media dopo ogni
  fascio.
- **Tempi**: linea temporale di trasferimenti, erogazioni e rotazioni del
  lettino.
- **Robot**: giunti, margini dai fine corsa e franco dagli ostacoli a ogni
  fascio.
- **Fasci**: tabella nell'ordine di erogazione; un clic porta il replay a quel
  fascio.

*Salva immagine* esporta la finestra in PNG. Tutti i grafici hanno un cursore
sincronizzato con il replay. Il resoconto si apre solo se è stato scritto dopo
l'avvio della simulazione: chiudendola prima dell'ultimo fascio, la scheda 4 lo
segnala invece di mostrare una seduta vecchia.

---

## Come funziona dentro

### Segmentazione del tumore

- **Rete**: UNet 3D (MONAI), patch 96³, ricampionamento a 1.5 × 1.5 × 2 mm,
  finestra HU da −1000 a +400. Inferenza a finestra scorrevole con fusione
  `constant` (la stessa usata in validazione).
- **Modello di produzione** (`outputs/model_final_all.pth`): pre-addestrato su
  **LIDC-IDRI** (221 serie di noduli con malignità ≥ 3.5) e
  **NSCLC-Radiomics** (415 serie con GTV disegnati da radioterapisti), poi
  addestrato su tutti i 63 casi MSD con *augmentation anatomica* (solo
  trasformazioni plausibili su un torace). I due dataset esterni sono
  complementari: LIDC porta lesioni piccole, NSCLC lesioni grandi.
- **Post-processing** (`postproc.py`): soglia di probabilità 0.5, lesioni
  sotto 100 mm³ scartate, la lesione più voluminosa definisce l'isocentro. È
  l'unica definizione, usata da validazione e applicazione.
- **Cancello di confidenza**: una lesione sotto 1 cm³ non viene considerata
  un bersaglio affidabile. L'errore della rete è bimodale — o entro pochi
  millimetri, o su un'altra regione del torace — e il cancello intercetta il
  secondo caso.

### Incertezza e bersaglio robusto

`uncertainty.py` ripete la predizione su 8 versioni perturbate della TAC e
misura probabilità media, deviazione standard, informazione mutua e, per ogni
campione, volume e centroide della lesione. Il segnale più utile è il **salto
massimo del centroide**: quando la rete sbaglia, in qualche versione
perturbata sceglie un'altra lesione. `robust_target.py` costruisce il **PTV
adattivo**: una corona a bassa probabilità attaccata alla lesione più un
margine di van Herk che somma in quadratura l'errore di contorno e quello di
moto.

### Anatomia

`organs.py` segmenta per densità (soglie Hounsfield e morfologia, nessuna
rete): polmoni destro e sinistro, vie aeree (cresciute da un seme sulla
trachea), scheletro (soglia a isteresi), profilo corporeo e mediastino
(ricostruito per criterio geometrico: il tessuto molle fra i polmoni, davanti
alla colonna). `organ_mesh.py` ricostruisce ogni struttura con una ricetta
propria (campo continuo dai valori HU, marching cubes, decimazione per
componente, lisciatura windowed-sinc, riparazione), con fedeltà media alla
superficie fisica di ~0.6 mm. La stessa anatomia serve al planner, alla vista
3D e a RViz.

### Dosimetria e pianificazione

- **Dose** (`dosimetry.py`): fotoni da 6 MV, densità elettronica ricavata
  dalle unità Hounsfield, TERMA lungo la profondità radiologica e una sola
  convoluzione FFT con un kernel di diffusione. Distanza sorgente-isocentro
  800 mm, collimatori circolari da 5 a 60 mm, griglia di calcolo a 1.5 mm.
- **Pianificatore inverso** (`planner.py`), in cinque stadi: fattibilità dei
  nodi del robot (lettino, inclinazioni, braccio) → riempimento del bersaglio
  con sfere che diventano punti di mira → candidati nodo × mira × cono →
  ottimo globale NNLS con penalità su spill, osso e percorso, poi
  sparsificazione → rifinitura su punti freddi e caldi. Nessun fascio supera il
  5% delle unità monitor. I parametri stanno tutti in `config.PlannerConfig`.
- **Il robot è dentro il piano**: prima della dose si ottimizzano base del
  robot e angoli del lettino, e l'ottimizzatore riceve solo i fasci che il
  braccio può eseguire. Così la dose calcolata è quella erogabile.

### Respiro e inseguimento

- `motion_model.py`: respiro sintetico con il modello di Lujan, irregolare,
  con isteresi e deriva; ampiezze dalla letteratura (Seppenwoolde 2002) in
  funzione della posizione della lesione. Simula la catena Synchrony: LED sul
  torace a 26 Hz, radiografie periodiche, modello di correlazione a due rami
  (inspirazione ≠ espirazione), predittori della latenza. Solo NumPy: gira
  anche nel nodo ROS.
- `motion_predictor.py`: predittore GRU addestrato su 300 respiri sintetici.
- `motion_dose.py`: dose vista dal tumore in movimento (approssimazione
  *static dose cloud*), ITV, margini di van Herk, attenuazione del fascio
  spostato.
- `tracking_session.py`: l'inseguimento dentro la seduta (latenza, cinematica
  d'inseguimento a 50 Hz, sospensione del fascio, correzioni del lettino).

### Robot

- **Manipolatore**: KUKA KR 300 R2500 ultra, lo stesso del CyberKnife M6/S7
  (quote dall'URDF pubblico di ROS-Industrial, portata 2500 mm). Cinematica
  inversa in forma chiusa; il grado di libertà in eccesso (rotazione attorno al
  fascio) si usa per stare lontani dai fine corsa.
- **Lettino robotico** (RoboCouch): un secondo manipolatore SCARA con polso
  roll-pitch-yaw, anch'esso in forma chiusa. Muovere il lettino equivale a
  muovere la base del robot al contrario, così gli ostacoli restano scatole
  allineate agli assi.
- **Collisioni**: braccio a capsule, lettino e paziente a scatole; si
  verificano anche i percorsi fra una posa e l'altra. Se un passaggio non è
  sicuro il braccio si ritira, ruota e riscende.
- **URDF** generato dalle costanti di `kinematics.py` (`urdf_gen.py`), mai
  scritto a mano: disegno e calcolo non possono divergere.

### Il ponte verso ROS 2

ROS 2 Humble gira in WSL perché `rclpy` non esiste per la versione di Python
dell'applicazione. I due mondi si scambiano file: il piano in JSON
(`plan_export.py`, millimetri, sistema RAS del paziente, con configurazione di
sala, respiro e dose sui voxel del tumore), l'anatomia in STL
(`export_scene.py`) e l'URDF. I moduli condivisi (`kinematics.py`,
`plan_export.py`, `dose_fill.py`, `treatment_log.py`, `motion_model.py`,
`tracking_session.py`) vengono copiati nel pacchetto da `urdf_gen.py`, e
`test_phase3.py` verifica che le copie coincidano con gli originali.

---

## Risultati principali

Misure **oneste**: cross-validation a 5 fold sui 63 casi MSD con etichetta;
ogni TAC è segmentata da un modello che non l'ha vista in addestramento.

**Segmentazione** (`outputs/eval/finetune_all_oof/`):

| Modello | Dice | Isocentro dentro il tumore | Falsi positivi per TAC |
|---|---:|---:|---:|
| storico | 0.451 | 73.0% | 0.95 |
| + augmentation anatomica | 0.470 | 81.0% | 0.67 |
| + pre-addestramento LIDC | 0.586 | 82.5% | 0.70 |
| **+ pre-addestramento LIDC + NSCLC** (produzione) | **0.623** (mediana 0.724) | **85.7%** | 0.75 |

Sensibilità per lesione 0.877. Con il cancello di confidenza il bersaglio è
corretto nel 94.1% dei casi accettati (81% dei casi).

**L'applicativo intero sui 63 casi** (15 ottobre 2026, report completo in
`archivio/documenti/`):

- 59 tumori trovati su 63, Dice mediano 0.77 sui trovati;
- semaforo: 49 verdi, 5 gialli, 9 rossi; **nei verdi l'isocentro è giusto in
  49 casi su 49**;
- tutti i 59 piani coprono il 95% del bersaglio pianificato, con il 100% dei
  fasci eseguibili dal robot e V20 polmonare mediano 0.5%;
- il tumore **vero** riceve la dose in mediana sul 71%: la segmentazione resta
  l'anello che decide quanta dose arriva.

**Tecniche di gestione del moto** sullo stesso paziente (lung_090, respiro di
11 mm, una seduta ROS per tecnica):

| Tecnica | Tumore alla dose prescritta | Volume irradiato a dose piena | Durata | Errore p95 |
|---|---:|---:|---:|---:|
| **Inseguimento** | **99.6%** | 7.1 cm³ | 216 s | 2.6 mm |
| Gating (finestra 3 mm) | 93.6% | 6.2 cm³ | 628 s | 3.9 mm |
| ITV, fascio fermo | 92.1% | 25.2 cm³ | 225 s | 6.6 mm |

**Predittori del respiro** (20 respiri mai visti): la GRU predice meglio
(RMSE 0.224 mm contro 0.266 del Kalman), ma l'errore d'inseguimento cambia
appena (p95 ~3.2 mm per tutti): domina il modello di correlazione che invecchia
fra una radiografia e l'altra. La leva è la frequenza delle radiografie.

Le misure di dettaglio, i tentativi scartati e la storia delle correzioni sono
in `archivio/documenti/README_precedente.md` e nei commenti in testa a ogni
modulo.

---

## Pipeline da riga di comando

L'applicativo usa modelli e calibrazioni già prodotti. Questi comandi servono a
rifarli. Tutti partono dalla radice del progetto.

### Dati

```bash
python splits.py                  # partizione a 5 fold stratificata per volume (outputs/splits_5fold.json)
python preprocess.py              # bounding box polmonari per caso
python lidc.py --index            # LIDC-IDRI: selezione delle serie
python lidc.py --convert          #           conversione DICOM + XML -> NIfTI
python nsclc.py --index           # NSCLC-Radiomics: inventario
python nsclc.py --convert --workers 6   #     conversione DICOM + RTSTRUCT -> NIfTI
```

### Addestramento del modello di produzione

```bash
# 1. pre-addestramento sui dati esterni (fold -1 = tutti i casi, senza validazione)
python train.py --exp pretrain_all --fold -1 --source lidc+nsclc --augment anatomic

# 2. cross-validation a 5 fold, per misurare le prestazioni (riprendibile, tetto di ore per sessione)
python run_cv.py --exp finetune_all --max-hours 7.5 -- --augment anatomic --init-from outputs/runs/pretrain_all/fold-1/final.pth
python run_cv.py --exp finetune_all --status

# 3. modello finale su tutti i 63 casi, poi copiato in outputs/model_final_all.pth
python train.py --exp final_all --fold -1 --augment anatomic --init-from outputs/runs/pretrain_all/fold-1/final.pth
```

`train.py` salva un checkpoint a ogni epoca e riparte con `--resume`; Ctrl-C
non perde l'epoca in corso.

### Valutazione

```bash
python evaluate.py --exp finetune_all --oof finetune_all --out outputs/eval/finetune_all_oof   # out-of-fold
python evaluate.py --exp finetune_all --out outputs/eval/finetune_all_oof --from-probs --sweep  # punto di lavoro, senza GPU
python gate.py --eval-dir outputs/eval/finetune_all_oof          # cancello di confidenza, validazione annidata
python compare.py --a outputs/eval/<esperimento_A> --b outputs/eval/<esperimento_B>   # confronto appaiato (Wilcoxon, McNemar)
```

### Incertezza, respiro e campagna

```bash
python run_uncertainty.py                 # mappe di incertezza dei 63 casi (~20 min GPU, riprendibile)
python run_uncertainty.py --analyze       # l'incertezza predice l'errore?
python robust_target.py --calibrate       # soglia della corona del PTV adattivo
python motion_predictor.py --train --benchmark   # predittore GRU e confronto (~2 min)
python campaign.py --n 20                 # pianificare sul predetto, giudicare sul vero (~2 ore, riprendibile)
```

L'esperimento su cui si basa la validazione dell'applicativo è
`config.UNC_EXP`: deve corrispondere al modello di produzione.

### Strumenti singoli

```bash
python inference.py --image <ct.nii.gz>              # segmentazione senza GUI
python tumor_3d.py --mask <seg.nii.gz>               # volume e mesh STL del tumore
python organ_mesh.py --ct <ct.nii.gz>                # qualità della ricostruzione degli organi
python organ_mesh.py --ct <ct.nii.gz> --tune         # ricerca delle ricette per struttura
python dosimetry.py --ct <ct.nii.gz> --mask <seg.nii.gz>   # dosimetria senza GUI
python plan_export.py --ct <ct.nii.gz> --mask <seg.nii.gz> --out ros2_ws/src/cyberknife_ros/plans/piano_trattamento.json
python export_scene.py --ct <ct.nii.gz> --mask <seg.nii.gz>   # anatomia in STL per RViz
python urdf_gen.py                                   # URDF + copie dei moduli nel pacchetto ROS
```

`urdf_gen.py` va rilanciato ogni volta che si modifica `kinematics.py` o uno
dei moduli condivisi con il nodo ROS.

---

## Test

Nessuno richiede ROS; i primi tre girano in pochi secondi.

```bash
python test_metrics_det.py      # metriche di detection su sfere sintetiche
python test_phase3.py           # robot, URDF, piano, copie nel pacchetto ROS, resoconto
python test_motion.py           # respiro, inseguimento, dose sotto moto
python test_uncertainty.py      # incertezza e bersaglio adattivo (--gpu: anche il campionamento)
python test_organ_mesh.py       # ricostruzione 3D (--sintetici: solo i fantocci)
python test_project.py          # salvataggio dei progetti (--gui: pipeline reale su lung_001)
```

---

## Installazione

### Requisiti

| Componente | Requisito |
|---|---|
| Python | 3.11 o successivo (testato su 3.14) |
| GPU | NVIDIA con CUDA consigliata; una serie 50xx (Blackwell) richiede PyTorch **cu128**. Senza GPU funziona, ma lentamente |
| VRAM | ~12 GB per l'addestramento (patch 96³, batch 2, AMP); molto meno per l'inferenza |
| Simulazione robot | WSL2 + Ubuntu 22.04 + ROS 2 Humble. **Facoltativa** |

### Ambiente Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Per GPU NVIDIA più vecchie usare l'indice `cu124` o `cu121`; senza GPU,
`pip install torch torchvision`. `config.py` contiene percorsi assoluti della
macchina di sviluppo (`OUTPUT_DIR`, `DATA_ROOT`): vanno adattati, oppure si
usano le variabili d'ambiente descritte in [Dataset](#dataset).

**Dopo ogni `git pull`** va rilanciato `pip install -r requirements.txt`: se
manca una libreria l'applicazione parte lo stesso, ma in modalità ridotta.

| Libreria | A cosa serve | Se manca |
|---|---|---|
| `torch`, `monai` | segmentazione, incertezza, predittore GRU | la percezione non parte |
| `PyQt6` | interfaccia | la GUI non parte |
| `pyvista`, `pyvistaqt` | vista 3D su GPU e geometria delle mesh | ripiego su Matplotlib: ~7 fps invece di ~200, mesh meno pulite; la barra di stato lo segnala |
| `scikit-image`, `trimesh`, `fast_simplification` | marching cubes, mesh, STL | la ricostruzione 3D non parte |
| `pydicom` (per `lidc.py` / `nsclc.py`) | conversione dei dataset esterni | solo il pre-addestramento ne risente |

L'elenco completo, con le note sulle versioni, è in
[`requirements.txt`](requirements.txt).

### ROS 2 per la simulazione (Fase 3)

Senza ROS la scheda 4 calcola comunque cinematica e grafici; manca solo la
simulazione visiva.

```bash
wsl --install -d Ubuntu-22.04
```

Poi, dentro WSL, l'installazione standard di ROS 2 Humble
(<https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html>) e:

```bash
sudo apt install ros-humble-desktop python3-colcon-common-extensions
```

`ros-humble-desktop` include RViz2, `robot_state_publisher`, `tf2_ros` e
`xacro`. Il pacchetto viene compilato da `ros2_ws/run.sh` in `~/ck_ws`, fuori
dalla cartella del progetto: il percorso `Progetto Medical` contiene uno spazio,
e il caricatore di mesh di RViz non accetta URL con spazi.

---

## Dataset

| Dataset | Uso | Cartella | Variabile d'ambiente |
|---|---|---|---|
| **MSD Task06 Lung** (Medical Segmentation Decathlon) | addestramento e validazione: 63 TAC con tumore etichettato, 32 senza | `Task06_Lung/Task06_Lung/{imagesTr,labelsTr,imagesTs}` | `MSD_LUNG_ROOT` |
| **LIDC-IDRI** (TCIA) | pre-addestramento, lesioni piccole | DICOM in `TCIA_LIDC-IDRI_20200921/`, annotazioni XML in `LIDC-XML-only/`, convertiti in `LIDC_nifti/` | `LIDC_ROOT`, `LIDC_XML_ROOT` |
| **NSCLC-Radiomics** (TCIA) | pre-addestramento, lesioni grandi (GTV clinici) | DICOM in `nsclc_radiomics/`, convertiti in `NSCLC_nifti/` | `NSCLC_ROOT` |

I dataset **non sono nel repository** (MSD da solo pesa ~8.6 GB). Per usare la
GUI basta una qualunque TAC del torace in `.nii.gz`; i dataset servono solo
per addestrare e validare.

```powershell
$env:MSD_LUNG_ROOT = "D:\percorso\Task06_Lung\Task06_Lung"
```

---

## Struttura del progetto

```
main.py               Avvio della GUI: finestra, menu, progetti, barra di stato
config.py             Configurazione unica: percorsi, modello, planner, post-processing, cancello

--- Interfaccia ---
ui_tabs.py            Scheda 1 (percezione) e contenitore della scheda 3
motion_tab.py         Scheda 2: respiro e tecnica di gestione del moto
viewer3d.py           Scheda 3: modello 3D e comandi della dosimetria
mesh_view_gpu.py      Vista 3D su GPU (PyVista/VTK)
section_view.py       Scheda 3: vista in sezione dei fasci
bev_view.py           Scheda 3: Beam's Eye View e DRR
dvh_view.py           Scheda 3: istogramma dose-volume con banda di incertezza
kinematics_tab.py     Scheda 4: cinematica, sala, avvio della simulazione ROS
report_view.py        Finestra del resoconto con il replay del tumore
inference_worker.py   Segmentazione e incertezza in background (QThread)
dose_worker.py        Dosimetria in background (QThread)
project_io.py         Salvataggio e apertura dei progetti .ckproj

--- Percezione ---
model.py              Architetture 3D (UNet, DynUNet, SegResNet)
dataset.py            Dati e trasformazioni MONAI, sorgenti MSD/LIDC/NSCLC
splits.py             Cross-validation a 5 fold, scritta una volta sola
preprocess.py         Bounding box polmonari
lidc.py, nsclc.py     Conversione dei dataset esterni in NIfTI
train.py              Addestramento di un fold, riprendibile
run_cv.py             Cross-validation completa con tetto di ore
predict.py            Inferenza -> mappa di probabilità nello spazio della TAC
inference.py          Inferenza da riga di comando
postproc.py           Post-processing unico e cancello di confidenza
evaluate.py           Valutazione out-of-fold con il post-processing di produzione
metrics_det.py        Metriche di detection (sensibilità, falsi positivi, errore di centroide)
gate.py               Taratura annidata del cancello
compare.py            Confronto statistico appaiato fra due esperimenti
uncertainty.py        Incertezza per TTA, semaforo
run_uncertainty.py    Mappe di incertezza dei 63 casi e loro analisi

--- Anatomia e dosimetria ---
organs.py             Segmentazione degli organi per densità
organ_mesh.py         Ricostruzione 3D per struttura e metriche di qualità
tumor_3d.py           Volume e mesh del tumore
dosimetry.py          Motore di dose (TERMA + convoluzione)
planner.py            Pianificatore inverso a cinque stadi
robust_target.py      PTV adattivo all'incertezza

--- Respiro ---
motion_model.py       Respiro, predittori, correlazione, inseguimento (solo NumPy)
motion_predictor.py   Predittore GRU
motion_dose.py        Dose sul tumore in movimento, ITV, margini
tracking_session.py   Inseguimento dentro la seduta (usato dal nodo ROS)
campaign.py           Campagna: pianificare sul predetto, giudicare sul vero

--- Robot e ROS 2 ---
kinematics.py         Cinematica del manipolatore e del lettino, collisioni, sala
urdf_gen.py           URDF dalle costanti di kinematics.py, copie nel pacchetto
plan_export.py        Contratto JSON fra dosimetria e ROS
export_scene.py       Anatomia in STL per RViz
dose_fill.py          Il tumore che si riempie di dose, fascio per fascio
treatment_log.py      Resoconto di fine seduta (condiviso con il nodo)
ros2_ws/              Workspace ROS 2: run.sh, build.sh, pacchetto cyberknife_ros
                      (nodo treatment_node.py, launch, URDF, mesh, piani, viste RViz)

--- Test ---
test_*.py             Vedi la sezione Test

outputs/              Modelli, risultati di valutazione, log, progetti salvati
archivio/             Documenti, presentazione, script una-tantum e backup (vedi archivio/README.md)
```
