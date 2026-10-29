"""Insert the partial-tying results table into srhm.tex (replaces the PT_TABLE placeholder)."""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
rs = [json.loads(l) for l in open(os.path.join(here, "exp_pt_results.jsonl"))]
get = {(r["rho"], r["L"], r["P"]): r for r in rs}
BS = chr(92)
lines = [BS + "small" + BS + "begin{tabular}{c|ccc|ccc}" + BS + "toprule",
         "& " + BS + "multicolumn{3}{c|}{$P=256$} & " + BS + "multicolumn{3}{c}{$P=1024$}" + BS + BS,
         "$" + BS + "rho$ & $L=6$ & $L=8$ & $L=10$ & $L=6$ & $L=8$ & $L=10$" + BS + BS + " " + BS + "midrule"]
for rho in [0.9, 0.8, 0.7, 0.6, 0.5]:
    cells = []
    for P in [256, 1024]:
        for L in [6, 8, 10]:
            r = get[(rho, L, P)]
            cells.append(f"{r['err']:.3f}/{r['floor']:.3f}")
    lines.append(f"{rho} & " + " & ".join(cells) + BS + BS)
lines.append(BS + "bottomrule" + BS + "end{tabular}")
tex_path = os.path.join(here, "..", "srhm.tex")
s = open(tex_path, encoding="utf-8").read()
s = s.replace("PT_TABLE", "\n".join(lines))
open(tex_path, "w", encoding="utf-8").write(s)
print("\n".join(lines))
