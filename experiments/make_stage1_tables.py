"""Fill the DIV2K matrix (and, when available, the context-rule table) into stage1.tex."""
import json, os
BS = chr(92)
here = os.path.dirname(os.path.abspath(__file__))
tex = os.path.join(here, "..", "stage1.tex")
s = open(tex, encoding="utf-8").read()
K = 6

r = json.load(open(os.path.join(here, "exp_selfsim_div2k.json")))
rows = [BS + "begin{center}" + BS + "small" + BS + "begin{tabular}{c|" + "c" * K + "}" + BS + "toprule",
        "from " + BS + " to & " + " & ".join(str(k) for k in range(K)) + BS + BS + " " + BS + "midrule"]
for k in range(K):
    cells = []
    for k2 in range(K):
        if k == k2:
            cells.append("--")
        else:
            a, b = r["sigma"][f"{k}->{k2}"][0], r["sigma_pink"][f"{k}->{k2}"][0]
            cells.append(f"{a:.2f} ({b:.2f})")
    rows.append(f"{k} & " + " & ".join(cells) + BS + BS)
rows.append(BS + "bottomrule" + BS + "end{tabular}" + BS + "end{center}")
s = s.replace("DIV2K_TABLE", "\n".join(rows))

ctx_path = os.path.join(here, "exp_selfsim_div2k_ctx.json")
if os.path.exists(ctx_path) and "CONTEXT_TABLE" in s:
    c = json.load(open(ctx_path))
    pairs = [p for p in c["context"]]
    t = [BS + "begin{center}" + BS + "small" + BS + "begin{tabular}{c|ccc|ccc}" + BS + "toprule",
         "& " + BS + "multicolumn{3}{c|}{parent only} & " + BS + "multicolumn{3}{c}{parent + context}" + BS + BS,
         "pair & $" + BS + "sigma$ & pink & own $R^2$ & $" + BS + "sigma$ & pink & own $R^2$" + BS + BS + " " + BS + "midrule"]
    for p in pairs:
        a, b = c["parent_only"][p], c["context"][p]
        t.append(f"${p.replace('->', BS + 'to ')}$ & {a['sigma'][0]:.2f} & {a['pink'][0]:.2f} & {a['r2_own']:.3f}"
                 f" & {b['sigma'][0]:.2f} & {b['pink'][0]:.2f} & {b['r2_own']:.3f}" + BS + BS)
    t.append(BS + "bottomrule" + BS + "end{tabular}" + BS + "end{center}")
    s = s.replace("CONTEXT_TABLE", "\n".join(t))
open(tex, "w", encoding="utf-8").write(s)
print("filled")
