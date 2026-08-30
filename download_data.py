from Bio import Entrez
import ssl
import os

# Omogućava pristup NCBI-u i kada lokalni SSL certifikat pravi problem
ssl._create_default_https_context = ssl._create_unverified_context

Entrez.email = "student@example.com"

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

sequences = {
    "frog_species_1": "M10217.1",
    "frog_species_2": "AY581667.1",
    "frog_species_3": "KM282504.1"
}

for species, accession in sequences.items():
    output_file = os.path.join(DATA_DIR, species + ".fasta")

    print(f"Downloading {accession}...")

    handle = Entrez.efetch(
        db="nucleotide",
        id=accession,
        rettype="fasta",
        retmode="text"
    )

    data = handle.read()
    handle.close()

    with open(output_file, "w") as f:
        f.write(data)

    print(f"Saved: {output_file}")

print("\nAll sequences downloaded successfully.")