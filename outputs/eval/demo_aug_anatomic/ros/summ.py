import json,sys
for c in sys.argv[1:]:
    d=json.load(open(f"{c}_resoconto.json",encoding="utf-8"));s=d["sintesi"];f=s["finale"]
    r=d.get("respiro") or {}
    print(c, f"fasci {s['fasci_erogati']}/{s['fasci_pianificati']}", f"CTV@Rx {f['copertura_pct']:.1f}% (pianif {s['copertura_pianificata_pct']:.1f})",
      f"D95 {f['d95_pct']:.1f} Dmin {f['dose_min_pct']:.1f}", f"durata {s['tempo_totale_s']:.0f}s", f"franco min {s['franco_minimo_mm']}mm margine {s['margine_minimo']}",
      {k:v for k,v in r.items() if not isinstance(v,(list,dict))})
