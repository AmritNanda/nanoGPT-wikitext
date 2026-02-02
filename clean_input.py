import re

print("Loading input.txt...")

with open("input2.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Cleaning wiki markup...")

# remove headers like == Track == OR = = Track = =
text = re.sub(r"(=+\s*.*?\s*=+)", "", text)

# remove @-@ artifacts
text = text.replace("@-@", "-")

# remove wiki links [[...]]
text = re.sub(r"\[\[.*?\]\]", "", text)

# remove templates {{...}}
text = re.sub(r"\{\{.*?\}\}", "", text)

# remove html tags <...>
text = re.sub(r"<.*?>", "", text)

# collapse multiple blank lines
text = re.sub(r"\n\s*\n", "\n\n", text)

with open("input_clean.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Done. Cleaned text saved as input_clean.txt")
