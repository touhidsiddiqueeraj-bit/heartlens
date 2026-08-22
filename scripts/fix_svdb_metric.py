#!/usr/bin/env python3
"""Fix SVDB metric: report N/V macro, not 3-class with APB=0 (brutal C3)."""
import json, pathlib
p = pathlib.Path("/home/touhid/heartlens/heart-lens-training/results/external_validation.json")
j = json.loads(p.read_text())
f1 = j["svdb_f1_per_class"]  # [N, APB, PVC]
n_f1, apb_f1, pvc_f1 = f1
nv_macro = (n_f1 + pvc_f1)/2
j["svdb_f1_normal"] = n_f1
j["svdb_f1_apb"] = apb_f1
j["svdb_f1_pvc"] = pvc_f1
j["svdb_macro_nv"] = nv_macro
j["svdb_macro_3class_biased"] = j["svdb_macro_f1"]
j["svdb_apb_support"] = 0
j["note"] = "APB absent in SVDB (0 windows) — 3-class macro is biased; report N/V macro"
# also count windows from external_validation log? hardcode from earlier: 28470 N, 1084 PVC
j["svdb_windows"] = {"Normal": 28470, "APB": 0, "PVC": 1084}
p.write_text(json.dumps(j, indent=2))
print(f"Fixed: N={n_f1:.4f} PVC={pvc_f1:.4f} N/V-macro={nv_macro:.4f} (biased 3-class {j['svdb_macro_3class_biased']:.4f})")
print(f"Wrote {p}")
