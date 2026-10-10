"""Clean reports, flag language, write the unique-string tables the translate/embed workers consume."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecglens.textprep import load_database

os.makedirs("data", exist_ok=True)
df = load_database("data/raw/ptbxl_database.csv")
print(df.lang.value_counts().to_string())
kept = df[df.lang != "sv"]
print(f"excluded {int((df.lang == 'sv').sum())} Swedish-flagged records; kept {len(kept)}")
# only German strings go through the German->English translator; English passes through unchanged
de_unique = sorted(kept.loc[kept.lang == "de", "report_clean"].unique())
json.dump(de_unique, open("data/de_unique.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"{len(de_unique)} unique German strings to translate; {kept.report_clean.nunique()} unique kept strings overall")
