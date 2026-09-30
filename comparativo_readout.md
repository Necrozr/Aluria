# Resultados do Comparativo: Readout de Broca vs Linear (TRU-Net)

**Ambiente:** `cuda` (NVIDIA GeForce RTX 2050)  
**Configuração:** Épocas: 3 | Seq: 32 | Batch: 16 | Símbolos: 16 | Vocab: 2000

## 1. Métricas Quantitativas

| Readout | Train Loss Final | Val Loss Final | Val Perplexity (PPL) | Tempo de Treino |
| :--- | :---: | :---: | :---: | :---: |
| **Linear (Baseline)** | 1.922 | 3.859 | **47.40** | 263.1s |
| **Broca (Proposto)** | -1.291 | 1.043 | **2.84** | 220.1s |

## 2. Exemplos de Geração de Texto

### Prompt: `A crase e obrigatoria`
- **Linear:** `<unk>ase <unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>`
- **Broca:** `<unk>ase <unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>`

### Prompt: `O sujeito da oracao`
- **Linear:** `<unk>jeito da ora<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>`
- **Broca:** `<unk>jeito da ora<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>`

### Prompt: `Concordancia verbal`
- **Linear:** `Conc<unk>ncia verbal <unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>`
- **Broca:** `Conc<unk>ncia verbal e Nominal e aponte a resposta correta.
0<unk><unk><unk><unk><unk><unk>`
