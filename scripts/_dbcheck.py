import sqlite3
c = sqlite3.connect("data/hybridmind.mind/store.db")
print("Total nodes:", c.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
rows = c.execute("SELECT json_extract(metadata, '$.session_id'), COUNT(*) FROM nodes GROUP BY 1").fetchall()
for sid, n in sorted(rows):
    print(f"  {sid}: {n}")
