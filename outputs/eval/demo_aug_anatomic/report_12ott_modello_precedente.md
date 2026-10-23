# CyberKnife Control Center — Risultati sull'intero dataset

Oct 12, 2026 · @nikke

## In sintesi

**Su tutte le 63 TAC il robot esegue sempre il piano, ma la dose arriva al tumore vero solo quanto è preciso il contorno disegnato dalla rete.** Il collo di bottiglia dell'applicativo è la segmentazione, non la dosimetria né il robot.

| Fase | Risultato su tutto il dataset |
| --- | --- |
| Percezione | 56 tumori trovati su 63; centro del tumore nel posto giusto in 52 casi su 56; precisione del contorno (Dice) mediana 0.60 |
| Semaforo | 41 verdi, 12 gialli, 10 rossi. Tra i verdi il centro è giusto in 40 casi su 41 |
| Dosimetria | Copertura pianificata 95% in tutti i 56 piani; polmone sano quasi intatto (V20 mediano 0.5%) |
| Robot | Il 100% dei fasci del piano finale è eseguibile in tutti i 56 casi |
| Tumore vero, piano sul solo contorno | Copertura mediana 49%: è qui che si perde |
| Respiro (20 pazienti) | Con il margine adattivo la copertura del tumore vero sale al 92%; con il contorno perfetto sarebbe 98% |
| ROS (6 sedute complete) | 6 su 6 con tutti i fasci erogati, 92–95% del bersaglio alla dose prescritta |
| Casi critici | 12 tumori toccano il mediastino; in 2 il mediastino riceve più della dose prescritta |

## Come è stato testato

Ogni TAC è stata trattata come un paziente nuovo, con lo stesso flusso dell'applicativo, e poi giudicata sul tumore disegnato dai medici.

- **Dataset**: Medical Segmentation Decathlon, Task06 Lung. 63 TAC con il contorno vero del tumore (volumi da 0.7 a 370 cm³, mediana 6.2). Le 32 TAC senza contorno non entrano nelle statistiche.
- **Contorno senza imbrogli (out-of-fold)**: validazione incrociata a 5 fold. Ogni TAC è segmentata dal modello del fold che non l'ha vista in addestramento, con la media delle 8 predizioni dell'analisi di incertezza.
- **Stesso flusso consegnato**: semaforo, ottimizzazione della sala, dosimetria con i soli fasci che il robot esegue. È il calcolo di `plan_export.py` e del pulsante della scheda 3.
- **Due giudizi diversi**: la copertura **pianificata** (sul contorno della rete, l'unico che l'applicativo conosce) e la copertura del **tumore vero** (sul contorno dei medici). Solo la seconda dice cosa riceverebbe il paziente.

Tre prove, dalla più ampia alla più completa:

| Prova | Pazienti | Che cosa include | Quando |
| --- | --- | --- | --- |
| Screening | 63 | Segmentazione, semaforo, sala, dosimetria, robot, distanze dagli organi | 7 ottobre 2026 |
| Campagna sul respiro | 20 (accettati dal cancello, scelti per volume) | Sei tecniche di gestione del moto con il tumore che respira | 18 settembre 2026 |
| Sedute ROS | 6 | Tutto, inclusa la simulazione completa del robot in ROS 2 con inseguimento | 7 ottobre 2026 |

Attenzione: il modello che usa l'applicativo in demo è stato addestrato su tutte e 63 le TAC, quindi su questi casi segmenta meglio di come farebbe su un paziente nuovo. I numeri di questo documento sono quelli onesti, out-of-fold.

## Percezione: trovare il tumore e sapere quando fidarsi

La rete trova un tumore in 56 TAC su 63 e ne mette il centro dentro il tumore vero in 52 casi. Il contorno è meno preciso del centro: Dice mediano 0.60 sui 56 casi trovati.

&#91;embedded content: outputs/eval/demo/screening\_completo.csv · 63 TAC, predizioni out-of-fold\]

- **Verde (41 casi)**: il centro è giusto in 40 casi su 41 (97.6%). L'unico errore è lung\_070, dove la rete sceglie con sicurezza una lesione a 220 mm da quella vera.
- **Giallo (12 casi)**: contorni nettamente peggiori; qui finiscono 2 dei 4 centri sbagliati. Il giallo serve proprio a dire «controlla a occhio».
- **Rosso (10 casi)**: in 7 la rete non trova nulla (lesioni da 0.8 a 8 cm³), negli altri 3 la lesione è sotto 0.5 cm³. Il semaforo non blocca: l'operatore può pianificare lo stesso.
- Il segnale che separa meglio i fallimenti è il salto massimo del centro fra le 8 predizioni perturbate (AUROC 0.93 sui 63 casi).

## Dove stanno i tumori

**I tumori che toccano il mediastino sono i più difficili per tutto il sistema**: più grandi, contornati peggio, coperti meno, e con più dose al cuore. Distanze misurate dal tumore vero, sui 56 casi pianificati.

| Distanza dal mediastino | Casi | Volume mediano | Dice mediano | Tumore vero coperto (mediana) | Dose massima al mediastino (mediana / peggiore) |
| --- | --- | --- | --- | --- | --- |
| Lontano (≥ 20 mm) | 35 | 4.9 cm³ | 0.68 | 57% | 10 / 57 |
| Vicino (2–20 mm) | 9 | 6.6 cm³ | 0.58 | 51% | 21 / 29 |
| A contatto (< 2 mm) | 12 | 30.7 cm³ | 0.44 | 30% | 54 / 124 |

Dosi sulla scala del piano, dove la prescrizione vale 80.

- **Vie aeree**: 4 tumori a meno di 10 mm (lung\_053, lung\_009, lung\_033, lung\_022). Solo in lung\_053 le vie aeree ricevono una dose alta (71).
- **Ossa**: 20 tumori su 56 stanno a meno di 2 mm da una costa o dalla colonna (parete toracica). In 5 casi l'osso vicino riceve dose vicina o oltre la prescrizione (fino a 101 su lung\_073): è il rischio di fratture costali noto in radiochirurgia.
- Il bordo fra tumore e mediastino ha la stessa densità in TAC: è il motivo per cui la rete ci sbaglia di più.

## Dosimetria: quanto promette il piano e quanto arriva davvero

La dosimetria fa sempre il suo lavoro sul contorno che riceve: tutti i 56 piani coprono il 93–95% del bersaglio. Ma il bersaglio è il contorno della rete, e il tumore vero ne riceve tanto quanto i due si sovrappongono.

&#91;embedded content: outputs/eval/demo/screening\_completo.csv · 56 piani, tumore fermo, piano sul solo contorno della rete\]

I punti salgono quasi in linea retta con la precisione del contorno: sotto un Dice di 0.5 nessun piano copre più di metà del tumore vero. Qui il piano è calcolato sul solo contorno, senza margini e con il tumore fermo: è il caso più severo, e la sezione sul respiro mostra quanto recuperano i margini.

**Qualità del piano e organi sani** (mediana e quartili sui 56 piani):

| Misura | Mediana | Intervallo centrale (25–75%) |
| --- | --- | --- |
| Conformità di Paddick sul tumore vero | 0.39 | 0.22–0.56 |
| Polmone che riceve almeno 20 Gy (V20) | 0.5% | 0.2–1.1% |
| Dose media al polmone | 0.8 Gy | 0.5–1.4 Gy |
| Dose massima al mediastino | 15 | 8–25 |
| Dose massima alle vie aeree | 5 | 2–13 |
| Dose massima alle ossa | 38 | 25–55 |

Dosi agli organi sulla scala del piano (prescrizione = 80). Il polmone sano è risparmiato in tutti i casi: il V20 più alto è 6.6% (limite clinico tipico 30–35%).

## Robot: il piano è sempre eseguibile

**In tutti i 56 piani il robot esegue il 100% dei fasci della dose finale**, perché il controllo del robot è dentro la dosimetria e non dopo.

| Misura | Risultato |
| --- | --- |
| Fasci nel piano finale | mediana 141 (da 41 a 150) |
| Piani in cui il collaudo ha scartato fasci prima della dose finale | 5 su 56 (52 fasci in tutto; il caso peggiore lung\_045 con 30) |
| Angoli di lettino usati | 3 in tutti i 56 casi |
| Tempo di calcolo per paziente (sala + dosimetria + collaudo) | mediana 4.4 min (da 2 a 6.7), 4 pazienti in parallelo |

- I piani con pochi fasci (41–98) sono quelli con tumori minuscoli (0.7–3.5 cm³): basta qualche cono piccolo.
- Il collaudo usa lo stesso solutore del nodo ROS: prima di questa scelta la GUI pianificava 107 fasci e il robot ne eseguiva 61, perdendo il 44% della dose (misurato su lung\_001). Ora pianificato ed erogato coincidono.
- Limite noto: il giunto A2 del KUKA ha solo 135° di corsa, e in circa 10 pose su 135 lavora vicino al fine corsa. Non fa perdere fasci, ma riduce il margine di sicurezza.

## Respiro: quale tecnica protegge davvero il tumore

Su 20 pazienti la campagna ha simulato il tumore che respira e confrontato sei modi di gestirlo, sempre pianificando sul contorno della rete e giudicando sul tumore vero.

&#91;embedded content: outputs/eval/campaign/summary.json · 20 pazienti, 6 tecniche, 18 settembre 2026\]

- **Il robot insegue bene**: rispetto al fascio fermo, l'inseguimento alza la copertura del tumore vero di 16 punti in mediana (p = 0.0002, test di Wilcoxon appaiato). L'errore residuo d'inseguimento è di circa 3 mm al 95° percentile, con radiografie ogni 60 s.
- **Ma non basta**: con il solo inseguimento il tumore vero resta coperto al 46%, perché il problema è il contorno e non il movimento.
- **Il margine adattivo** allarga il bersaglio dove la rete è incerta: 92% di copertura, quasi come l'ITV (93%), con un volume trattato minore (21 contro 29 cm³). Su 20 casi la differenza di volume con l'ITV non è statisticamente significativa (p = 0.43); sulle lesioni sotto 10 cm³ il volume scende a 0.59 volte.
- **Con un contorno perfetto** il sistema arriverebbe al 98% con soli 11 cm³: è il margine di miglioramento ancora disponibile, ed è tutto nella segmentazione.
- Il polmone sano resta poco irradiato in ogni tecnica: V20 mediano da 0.5% (inseguimento) a 2.3% (ITV).

## Simulazione ROS: sei sedute complete

**In tutte e 6 le sedute il robot ha erogato ogni fascio e consegnato quasi esattamente la dose pianificata.** I quattro casi buoni restano buoni anche sul tumore vero; i due critici no, e la seduta ROS da sola non se ne accorge.

Tecnica: inseguimento con margine di van Herk, predittore Kalman, radiografie ogni 60 s, 120 s di fascio acceso, simulazione headless in ROS 2 Humble.

| Caso | Tumore | Fasci erogati | Bersaglio alla dose in ROS (pianificato) | Tumore vero coperto, con il respiro | Dose massima al mediastino | Durata | Fascio sospeso |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lung\_005 | 7 cm³, respiro 14 mm | 139/139 | **95.4%** (95.0%) | **97.5%** | 26 | 213 s | 0% |
| lung\_025 | 4 cm³ | 143/143 | 95.4% (95.0%) | 98.7% | 14 | 223 s | 0.1% |
| lung\_014 | 31 cm³ | 147/147 | 94.4% (94.8%) | 99.5% | 9 | 221 s | 0.6% |
| lung\_062 | 5 cm³ | 147/147 | 92.2% (95.0%) | 99.2% | 10 | 226 s | 1.3% |
| lung\_009 | 89 cm³, centrale | 144/144 | 92.6% (95.2%) | **46%** | **103** | 212 s | 0% |
| lung\_053 | 370 cm³, centrale | 145/145 | 92.0% (95.0%) | **16%** | **112** | 220 s | 0% |

Dosi sulla scala del piano (prescrizione = 80). «Bersaglio in ROS» è giudicato sul contorno della rete, come fa il resoconto dell'applicativo; «tumore vero» è calcolato a parte sul contorno dei medici.

- La perdita fra pianificato ed erogato è al massimo di 3 punti: è l'errore residuo dell'inseguimento (p95 circa 3 mm).
- Il franco minimo del braccio dagli ostacoli va da 5 a 16 mm, tranne lung\_025 (0.5 mm).
- Nei log compaiono avvisi *Moved backwards in time*: l'orologio di WSL salta all'indietro. Non toccano dose e resoconto, ma in RViz il braccio può fare piccoli scatti.

## Dove l'applicativo fallisce

**I fallimenti si concentrano agli estremi: tumori minuscoli, che la rete non vede, e tumori enormi o centrali, che la rete vede solo in parte.** Le lesioni medie, da 2 a 50 cm³, sono quelle dove il sistema funziona meglio.

| Volume del tumore vero | Casi pianificati | Dice mediano | Tumore vero coperto (mediana) |
| --- | --- | --- | --- |
| Sotto 2 cm³ | 7 (+6 non trovati) | 0.52 | 37% |
| 2–10 cm³ | 27 (+1 non trovato) | 0.68 | 56% |
| 10–50 cm³ | 15 | 0.76 | 65% |
| Oltre 50 cm³ | 7 | 0.27 | 17% |

I quattro tipi di fallimento osservati:

| Tipo | Casi | Esempio | Lo segnala il semaforo? |
| --- | --- | --- | --- |
| Nessun tumore trovato | 7 (lesioni da 0.8 a 8 cm³) | lung\_036 (8 cm³) | Sì, rosso: è un rifiuto corretto |
| Bersaglio sbagliato (centro fuori dal tumore) | 4 | lung\_070, lesione scelta a 220 mm da quella vera | In 3 casi su 4 (2 gialli, 1 rosso); **lung\_070 è verde** |
| Tumore grande visto solo in parte | 7 oltre 50 cm³ | lung\_028 (151 cm³, Dice 0.01), lung\_053 (370 cm³, Dice 0.17) | Spesso giallo, non sempre |
| Tumore centrale attaccato al mediastino | 12 a contatto | lung\_009: mediastino a 103, tumore vero coperto al 46% | **No: verde** |

Perché il caso centrale è il più insidioso:

- in TAC tumore e mediastino hanno la stessa densità, quindi la rete non trova il bordo;
- il piano, conforme al contorno sbagliato, porta la dose ablativa dentro il cuore e i grossi vasi;
- robot e ROS eseguono tutto perfettamente, e il resoconto (giudicato sul contorno della rete) dice 92%;
- clinicamente, per i tumori «ultracentrali» la radiochirurgia ad alte dosi è rischiosa: è un caso in cui la tecnica stessa non è la scelta migliore.

Altri limiti misurati:

- Il respiro è simulato: il dataset non contiene TAC 4D.
- La segmentazione degli organi è a soglie: il mediastino non separa cuore, vasi ed esofago, e su alcuni pazienti include il fegato.
- Il modello consegnato è addestrato su tutte le 63 TAC: per una valutazione cieca dal vivo servono TAC nuove (le 32 di test non hanno il contorno vero).

## Cosa dicono i dati sull'applicativo

**La catena a valle della segmentazione è affidabile; la segmentazione è il punto debole, e i margini adattivi ne recuperano gran parte.**

1. **Dosimetria e robot sono solidi.** 56 piani su 56 eseguibili al 100%, 6 sedute ROS su 6 complete, al massimo 3 punti persi fra pianificato ed erogato, polmone sano risparmiato ovunque.
2. **Il semaforo funziona come filtro.** Tra i casi verdi il centro del tumore è giusto in 40 su 41, e i tumori non trovati sono tutti rossi. Non vede però i casi centrali, dove la rete è sicura ma sbaglia il bordo.
3. **Il contorno decide tutto.** Il tumore vero riceve la dose in proporzione al Dice; con un contorno perfetto la stessa catena coprirebbe il 98%.
4. **I margini compensano.** Il margine adattivo all'incertezza porta la copertura del tumore vero dal 46% al 92%, con meno volume trattato dell'ITV.
5. **Funziona meglio sulle lesioni medie e periferiche** (2–50 cm³, lontane dal mediastino), che sono anche quelle per cui la radiochirurgia robotica è indicata.

Prossimi passi suggeriti dai dati:

- Migliorare la segmentazione dei tumori centrali e grandi: è la leva più grande rimasta.
- Aggiungere al semaforo un avviso di vicinanza al mediastino (12 casi su 56 a contatto), che oggi resta verde.
- Mostrare nel resoconto un avviso quando la dose al mediastino supera la prescrizione.
- Ridurre gli scatti in RViz legati all'orologio di WSL.

Dati completi: `outputs/eval/demo/screening_completo.csv` (63 casi), `outputs/eval/campaign/` (campagna sul respiro), `outputs/eval/demo/ros/` (sedute ROS e valutazioni sul tumore vero).
