import sys
content = sys.stdin.read()
with open("explore_premium.py", "w", encoding="utf-8") as out:
    out.write(content)
print(f"Written {len(content)} bytes")
