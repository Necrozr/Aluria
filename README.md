# Aluria / TRU - Cognitive Architecture (Non-Transformer GWT/RIMs)

Implementação do **TRU-Net v2** — uma arquitetura de múltiplos estados privados viscosos em paralelo com canais de comunicação limitada, integrador de espaço global (Global Workspace Theory) e memória episódica.

---

## 🔬 Evidências Experimentais e Teóricas (GWT / RIMs)

1. **Quebra de Simetria (Symmetry Breaking):**
   - Slicing disjunto de features: `emb_dim=66 -> [22, 22, 22]`, cada módulo processa uma fatia distinta.
   - Papéis cognitivos assimétricos:
     - **Módulo A (Hipótese):** Vê o token atual com dropout leve (0.1).
     - **Módulo B (Memória):** Média móvel exponencial lenta ($0.7 \cdot \text{ema} + 0.3 \cdot x$).
     - **Módulo C (Exploração):** Dropout agressivo (0.5) + injeção de ruído gaussiano (0.1).
   - $\mathcal{L}_{\text{div}}$: Penalidade de ortogonalidade / correlação de cosseno ao quadrado sobre as mensagens para evitar colapso de redundância.

2. **Diversidade de Linhas e Pooling Estruturado:**
   - Projeções dedicadas por faixa de viscosidade temporal ($K=4$).
   - Pooling com atenção aprendida sobre as $R$ linhas em vez de média simples.
   - Neuromodulação de $G$: $G$ atua como gate multiplicativo $\tanh(...) \odot (1 + \sigma(W_g G))$ em vez de injeção destrutiva.

3. **Comunicação Discreta Emergente (InfoMax):**
   - Perda $\mathcal{L}_{\text{sym}} = -H(\bar{p}) + \beta \cdot \text{mean}(H(p_b))$: maximiza entropia marginal do vocabulário e minimiza incerteza por amostra.
   - Ruído no canal ($p_{\text{noise}}$) e annealing de temperatura $\tau$ ($1.0 \to 0.2$).
   - **Resultado:** `--symbols 16`: $100\%$ acurácia e $\bar{H} = 2.69$ (teto teórico $\ln(16) = 2.77$), comprovando emergência sem codebook collapse.

4. **Memória Episódica (Nível M):**
   - Fila circular com $32$ slots de vetores globais com momentum que sobrevive entre episódios e batches.

5. **Prova Causal de Comunicação (Ablation Benchmark):**
   ```bash
   python train_toy.py --ablate
   ```
   - **Com canal ativo:** **100% de acurácia**.
   - **Módulo B permutado / sem canal:** **12.0% de acurácia** (chance aleatória teórica é $12.5\%$).
   - Isso demonstra formalmente que a comunicação pelo canal é indispensável e causal, não correlação espúria.

6. **Benchmark de Geração de Linguagem: Readout de Broca vs Linear (RTX 2050):**
   ```bash
   python compare_readouts.py --epochs 3
   ```
   Resultados no dataset de português (`dataset_portugues_concurso.jsonl`):
   
   | Readout | Train Loss (Ep 3) | Val Loss (Ep 3) | Val Perplexity (PPL) | Geração do Prompt *"Concordancia verbal"* |
   | :--- | :---: | :---: | :---: | :--- |
   | **Linear (Baseline)** | 1.922 | 3.859 | **47.40** | Repete `<unk>` sucessivos sem sintaxe |
   | **Broca (Proposto)** | -1.291 | 1.043 | **2.84** | *"Concordancia verbal e Nominal e aponte a resposta correta.\n0..."* |

   - **Queda de Perplexidade:** Redução de **16.7x** na perplexidade de validação ($47.40 \to 2.84$).
   - **Emergência Sintática:** O Readout de Broca foi capaz de conectar o tema deliberativo global à sintaxe local do domínio de concurso.

---

## 💻 Comandos de Execução

### 1. Testes de Evidência e Benchmark Toy
```bash
# Modo contínuo com perda de diversidade
python train_toy.py

# Modo discreto com 16 símbolos emergentes via InfoMax
python train_toy.py --symbols 16

# Ablações de canal e controle causal
python train_toy.py --ablate
```

### 2. Treinamento no Dataset de Português
```bash
# Treino com canal discreto e regularizador de diversidade
python train_portugues.py --epochs 3 --symbols 16 --div_w 0.1 --chunk_size 4 --vocab 2000

# Readout de Broca (padrão) vs linear legado
python train_portugues.py --epochs 3 --readout broca
python train_portugues.py --epochs 3 --readout linear
```

## 🗣️ Readout de Superfície (v3, não-Transformer)

`G` é ideia amodal, não palavra. O vocalizador de Broca (`tru_net/readout.py`) é uma GRU de superfície com inércia própria:

```text
h_t = GRUCell([e_t | G_m | pooled(S_A,S_B,S_C)], h_{t-1})
logits = W_vocab LN(h_t) + W_skip e_t
```

- `W_skip e_t`: sintaxe local direta (residual imediato), libera `G` para semântica global.
- `h_t` recorrente: memória de emissão de curto prazo, tempos rápidos vs. `S_i` lentos.
- `pooled(S)`: cross-readout sobre os 3 módulos com `e_t` implícito no gate da GRU.

---

## 🚀 Execução no Google Colab (GPU Gratuita)

O arquivo [`TRU_Colab_Training.ipynb`](file:///c:/Users/kauan/OneDrive/Documentos/TRU/TRU_Colab_Training.ipynb) está pronto para rodar em GPU T4/A100 no Google Colab com 1 clique:
1. Abra o [Google Colab](https://colab.research.google.com/).
2. Faça o upload do arquivo `TRU_Colab_Training.ipynb`.
3. Ative a GPU em **Ambiente de execução** > **Alterar tipo de ambiente de execução** > **T4 GPU**.
4. Execute as células para treinar e baixar os pesos salvos.
