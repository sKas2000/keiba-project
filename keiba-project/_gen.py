import pathlib

SQ = chr(39)
DQ = chr(34)
TQ = DQ * 3
NL = chr(10)
BS = chr(92)

# Read template
tmpl = pathlib.Path("_template.txt").read_text(encoding="utf-8")

# Replace placeholders
tmpl = tmpl.replace("__SQ__", SQ)
tmpl = tmpl.replace("__TQ__", TQ)

# Write output
pathlib.Path("explore_premium.py").write_text(tmpl, encoding="utf-8")
print(f"Written {len(tmpl)} bytes")