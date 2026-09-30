"""
Exemplo de carregamento e inspeção do dataset em Python.
Compatível com Pandas, Hugging Face Datasets e Fine-Tuning.
"""

import json
import pandas as pd

# 1. Carregar via JSONL nativo
print("--- LENDO COM PYTHON PURO ---")
with open("dataset_portugues_concurso.jsonl", "r", encoding="utf-8") as f:
    primeiro_registro = json.loads(f.readline())
    print("ID:", primeiro_registro["id"])
    print("Tipo:", primeiro_registro["type"])
    print("Tópico:", primeiro_registro["topic"])
    print("Instrução:", primeiro_registro["instruction"][:80], "...")

# 2. Carregar com Pandas
print("\n--- LENDO COM PANDAS ---")
df = pd.read_json("dataset_portugues_concurso.jsonl", lines=True)
print(f"Total de linhas carregadas: {len(df)}")
print("Distribuição por tipo:")
print(df["type"].value_counts())

# Exemplo de como usar no Hugging Face:
# from datasets import load_dataset
# ds = load_dataset("json", data_files="dataset_portugues_concurso.jsonl")
