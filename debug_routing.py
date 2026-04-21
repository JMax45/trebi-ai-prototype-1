import pathlib, json

SKILLS_DIR = pathlib.Path("skills")
skills = {}
for f in SKILLS_DIR.glob("*.json"):
    data = json.loads(f.read_text(encoding="utf-8"))
    skills[data["name"]] = data
    print(f"Loaded: {data['name']} ({f.name})")

print()
print("Iteration order:", list(skills.keys()))

# Check exactly which reporting keywords match a mixed question
q = "scrivi un rapportino per intervento di sostituzione componente guasto"
print(f"\nQuestion: '{q}'")
for name, data in skills.items():
    hits = [(kw, kw in q) for kw in data.get("keywords", [])]
    matched = [kw for kw, ok in hits if ok]
    print(f"  Skill '{name}': {matched if matched else '(no match)'}")

# Byte-level check for 'rapportino'
rep_kws = skills.get("reporting", {}).get("keywords", [])
print()
print("Reporting keyword repr list:")
for kw in rep_kws:
    hit = kw in q
    print(f"  {repr(kw)} -> {hit}")

# Simulate route_skill
print()
for name, data in skills.items():
    if name == "general":
        continue
    if any(kw in q for kw in data.get("keywords", [])):
        print(f"route_skill returns: '{name}', tools: {data.get('tools')}")
        print(f"  write_file available: {'write_file' in data.get('tools', [])}")
        break
