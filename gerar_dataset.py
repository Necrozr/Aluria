"""
Script para extrair e estruturar a Apostila de Português para Concursos
em um dataset para IA no formato JSONL (Instrução / Q&A / Treinamento de LLMs).
"""

import sys
import os
import json
import re
import pymupdf

sys.stdout.reconfigure(encoding='utf-8')

PDF_PATH = "apostila-portugues-para-concurso.pdf"
OUTPUT_JSONL = "dataset_portugues_concurso.jsonl"
SYSTEM_PROMPT = "Você é um professor especialista em Língua Portuguesa para concursos públicos."

def clean_page(page):
    """Remove cabeçalhos e numeração de página repetitivos."""
    lines = page.get_text().split('\n')
    cleaned = []
    for l in lines:
        s = l.strip()
        if not s:
            continue
        if s in ['Polícia Rodoviária Federal', 'Apostila de Português para Concursos']:
            continue
        if re.match(r'^\d{1,3}$', s):
            continue
        cleaned.append(s)
    return '\n'.join(cleaned)

def extract_concurso_batteries(doc):
    """
    Extrai as 20 baterias de questões de concurso (páginas 215 a 238)
    com suas 200 questões e gabaritos oficiais.
    """
    full_text = ""
    for p in range(214, 238):
        full_text += '\n' + clean_page(doc[p])

    parts = re.split(r'(?:R:\s*)?GABARITO', full_text, flags=re.IGNORECASE)
    items = []

    topic_keywords = [
        ("concordância verbal", "Sintaxe - Concordância Verbal"),
        ("concordância nominal", "Sintaxe - Concordância Nominal"),
        ("concordância", "Sintaxe - Concordância"),
        ("regência nominal", "Sintaxe - Regência Nominal"),
        ("regência verbal", "Sintaxe - Regência Verbal"),
        ("regência", "Sintaxe - Regência"),
        ("crase", "Sintaxe - Crase"),
        ("pontuação", "Pontuação"),
        ("vírgula", "Pontuação"),
        ("ortografia", "Ortografia"),
        ("grafada", "Ortografia"),
        ("grafia", "Ortografia"),
        ("fonema", "Fonética"),
        ("sílaba", "Fonética - Divisão Silábica"),
        ("oxítona", "Acentuação Gráfica"),
        ("paroxítona", "Acentuação Gráfica"),
        ("proparoxítona", "Acentuação Gráfica"),
        ("acento", "Acentuação Gráfica"),
        ("acentuada", "Acentuação Gráfica"),
        ("acentuado", "Acentuação Gráfica"),
        ("plural", "Morfologia - Substantivo (Flexão de Plural)"),
        ("gênero", "Morfologia - Substantivo (Flexão de Gênero)"),
        ("substantivo", "Morfologia - Substantivo"),
        ("pronome", "Morfologia - Pronomes"),
        ("verbo", "Morfologia - Verbos"),
        ("adjetivo", "Morfologia - Adjetivo"),
        ("advérbio", "Morfologia - Advérbio"),
        ("numeral", "Morfologia - Numeral"),
        ("figura", "Estilística - Figuras de Linguagem"),
        ("hífen", "Ortografia - Emprego do Hífen"),
        ("porquê", "Ortografia - Uso dos Porquês"),
        ("por que", "Ortografia - Uso dos Porquês"),
        ("por quê", "Ortografia - Uso dos Porquês"),
        ("sujeito", "Sintaxe - Termos da Oração (Sujeito)"),
        ("predicado", "Sintaxe - Termos da Oração (Predicado)"),
        ("oração", "Sintaxe do Período Composto"),
        ("período", "Sintaxe do Período Composto"),
        ("voz do verbo", "Morfologia - Vozes Verbais"),
        ("voz passiva", "Morfologia - Vozes Verbais"),
        ("mórfico", "Morfologia - Estrutura das Palavras"),
        ("radical", "Morfologia - Estrutura das Palavras"),
        ("derivação", "Morfologia - Formação das Palavras"),
        ("composição", "Morfologia - Formação das Palavras"),
        ("significado", "Semântica - Significação das Palavras"),
        ("sentidos foram trocados", "Semântica - Significação das Palavras")
    ]

    def infer_topic(text):
        low = text.lower()
        for kw, top in topic_keywords:
            if kw in low:
                return top
        return "Língua Portuguesa para Concursos"

    for idx in range(len(parts)-1):
        q_section = parts[idx]
        if idx == 0 and 'EXERCÍCIOS' in q_section:
            q_section = q_section.split('EXERCÍCIOS', 1)[1]

        next_part = parts[idx+1]
        gab_lines = [l for l in next_part.strip().split('\n') if l.strip()][:3]
        gab_text = ' '.join(gab_lines)
        answers = dict(re.findall(r'(\d{1,2})\s*[-–\.:]\s*([A-E])', gab_text))

        if idx > 0:
            lines = q_section.strip().split('\n')
            q_start = 0
            while q_start < len(lines) and ('|' in lines[q_start] or re.search(r'\d{2}\s*[-–]\s*[A-E]', lines[q_start])):
                q_start += 1
            q_section = '\n'.join(lines[q_start:])

        raw_qs = re.split(r'\n(?=\d{2}(?:\s*[\.\-–]|\s+|[A-ZÀ-Ú]))', '\n' + q_section.strip())
        qs = [q.strip() for q in raw_qs if q.strip() and re.match(r'^\d{2}(?:\s*[\.\-–]|\s+|[A-ZÀ-Ú])', q.strip())]

        for q in qs:
            m = re.match(r'^(\d{2})', q)
            qnum = m.group(1) if m else "00"
            correct_opt = answers.get(qnum, answers.get(str(int(qnum)), "Gabarito na apostila"))
            topic = infer_topic(q)

            # Separa enunciado e opções se possível
            prompt_user = f"Resolva a seguinte questão de concurso sobre {topic}:\n\n{q}"
            response_text = f"**Alternativa correta:** {correct_opt}\n\n**Tema avaliado:** {topic}.\nEsta questão avalia os conhecimentos práticos e a aplicação da norma culta da Língua Portuguesa cobrada em concursos públicos."

            item = {
                "id": f"concurso_bat{idx+1:02d}_q{qnum}",
                "type": "exercicio",
                "source": "Apostila de Português para Concursos - Baterias Finais",
                "battery": idx + 1,
                "question_number": int(qnum),
                "topic": topic,
                "instruction": f"Resolva a questão de concurso sobre {topic} e identifique a alternativa correta com base na norma culta.",
                "input": q,
                "output": response_text,
                "gabarito": correct_opt,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_user},
                    {"role": "assistant", "content": response_text}
                ]
            }
            items.append(item)

    return items

def extract_chapter_exercises(doc):
    """
    Extrai exercícios e gabaritos de fixação presentes nos capítulos teóricos.
    """
    chapter_specs = [
        {"name": "Fonética", "pages": (8, 15), "resp_page": 14, "num_q": 14},
        {"name": "Separação Silábica", "pages": (15, 18), "resp_page": 17, "num_q": 8},
        {"name": "Acentuação Gráfica", "pages": (18, 24), "resp_page": 22, "num_q": 14},
        {"name": "Ortografia", "pages": (24, 31), "resp_page": 29, "num_q": 14},
        {"name": "Significação das Palavras", "pages": (30, 33), "resp_page": 32, "num_q": 5},
        {"name": "Emprego do Hífen", "pages": (33, 36), "resp_page": 34, "num_q": 11},
        {"name": "Uso dos Porquês", "pages": (36, 38), "resp_page": 37, "num_q": 14},
        {"name": "Estrutura e Formação das Palavras", "pages": (38, 44), "resp_page": 42, "num_q": 14},
        {"name": "Morfologia - Substantivo", "pages": (43, 58), "resp_page": 57, "num_q": 5},
        {"name": "Morfologia - Numeral", "pages": (58, 61), "resp_page": 60, "num_q": 5},
        {"name": "Morfologia - Adjetivo", "pages": (61, 69), "resp_page": 68, "num_q": 1},
        {"name": "Concordância Nominal", "pages": (70, 75), "resp_page": 74, "num_q": 5},
        {"name": "Colocação Pronominal", "pages": (91, 95), "resp_page": 93, "num_q": 5},
        {"name": "Morfologia - Verbos", "pages": (101, 117), "resp_page": 115, "num_q": 5},
        {"name": "Predicação Verbal", "pages": (121, 123), "resp_page": 121, "num_q": 5},
        {"name": "Regências Verbal e Nominal", "pages": (124, 132), "resp_page": 130, "num_q": 5},
        {"name": "Concordância Verbal", "pages": (139, 146), "resp_page": 144, "num_q": 5},
        {"name": "A palavra SE", "pages": (166, 172), "resp_page": 170, "num_q": 5},
        {"name": "Crase", "pages": (172, 178), "resp_page": 176, "num_q": 5},
        {"name": "Período Composto", "pages": (182, 185), "resp_page": 183, "num_q": 5}
    ]

    items = []
    for spec in chapter_specs:
        resp_p = spec["resp_page"]
        text_resp = clean_page(doc[resp_p])
        ans_matches = re.findall(r'(\d{1,2})\s*[-–\.\)]\s*([A-Ea-eCIci])', text_resp)
        answers = {str(int(m[0])): m[1].upper() for m in ans_matches}

        # Concatena páginas do capítulo para encontrar os enunciados
        cap_text = ""
        for p in range(spec["pages"][0], spec["pages"][1] + 1):
            cap_text += "\n" + clean_page(doc[p])

        # Procura seção de exercícios
        m_ex = re.search(r'EXERCÍCIOS?\b', cap_text, flags=re.IGNORECASE)
        if m_ex:
            ex_body = cap_text[m_ex.start():]
            m_resp = re.search(r'Respostas\b', ex_body, flags=re.IGNORECASE)
            if m_resp:
                ex_body = ex_body[:m_resp.start()]

            # Divide questões por números como 1), 1-, 01, 1.
            raw_qs = re.split(r'\n(?=\d{1,2}\s*[\.\-\)]\s*)', ex_body)
            qs = [q.strip() for q in raw_qs if q.strip() and re.match(r'^\d{1,2}\s*[\.\-\)]', q.strip())]

            for q in qs:
                num_match = re.match(r'^(\d{1,2})', q)
                if not num_match:
                    continue
                num_str = str(int(num_match.group(1)))
                ans = answers.get(num_str)
                if not ans:
                    continue

                topic = spec["name"]
                ans_desc = f"Alternativa correta: **{ans}**" if ans in ['A','B','C','D','E'] else f"Classificação: **{ans}** (C = Certo / I = Incorreto)"
                output_str = f"{ans_desc}.\n\n**Fundamentação teórica:** Exercício prático de fixação sobre o tema **{topic}** conforme as regras gramaticais da apostila."

                item = {
                    "id": f"cap_ex_{topic.lower()[:4]}_q{num_str}",
                    "type": "exercicio",
                    "source": f"Apostila de Português - Exercícios de Fixação: {topic}",
                    "topic": topic,
                    "instruction": f"Resolva o seguinte exercício de concurso sobre {topic} e aponte a resposta correta.",
                    "input": q,
                    "output": output_str,
                    "gabarito": ans,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Resolva o seguinte exercício sobre {topic}:\n\n{q}"},
                        {"role": "assistant", "content": output_str}
                    ]
                }
                items.append(item)

    return items

def extract_theory_units(doc):
    """
    Extrai as seções teóricas da apostila e as transforma em unidades instrucionais
    de Pergunta / Resposta Didática (Q&A e Fine-Tuning).
    """
    theory_topics = [
        {"title": "Interpretação de Textos", "range": (3, 7),
         "instruction": "Como desenvolver uma interpretação de texto eficiente para concursos públicos e quais são as principais armadilhas em questões interpretativas?"},
        {"title": "Fonética - Conceitos Fundamentais, Encontros Vocálicos e Consonantais", "range": (8, 13),
         "instruction": "Explique a diferença entre ditongo, tritongo e hiato, e como identificar encontros consonantais e dígrafos na Língua Portuguesa."},
        {"title": "Separação Silábica e Regras de Translineação", "range": (14, 17),
         "instruction": "Quais são as regras fundamentais de divisão silábica e como realizar a translineação correta ao final de linhas segundo a norma culta?"},
        {"title": "Regras Gerais e Especiais de Acentuação Gráfica", "range": (17, 22),
         "instruction": "Apresente as regras de acentuação gráfica para palavras oxítonas, paroxítonas, proparoxítonas, monossílabos tônicos e hiatos."},
        {"title": "Ortografia Oficial - Emprego de Letras e Grafias Duvidosas", "range": (23, 29),
         "instruction": "Explique as regras práticas para o uso correto das letras S, Z, X, CH, J e G na Língua Portuguesa com exemplos."},
        {"title": "Significação das Palavras - Sinonímia, Antonímia, Homonímia e Paronímia", "range": (29, 32),
         "instruction": "Diferencie homônimos homógrafos, homófonos e parônimos, citando exemplos frequentes em provas de concursos públicos."},
        {"title": "Emprego do Hífen", "range": (32, 35),
         "instruction": "Explique as regras do emprego do hífen em palavras compostas e com prefixos na Língua Portuguesa."},
        {"title": "Emprego dos Porquês (Por que, Por quê, Porque, Porquê)", "range": (35, 37),
         "instruction": "Explique detalhadamente quando utilizar 'por que', 'por quê', 'porque' e 'porquê', fornecendo frases exemplares para cada caso."},
        {"title": "Estrutura e Morfologia das Palavras", "range": (37, 39),
         "instruction": "Quais são os elementos mórficos que compõem uma palavra (radical, tema, desinências, afixos e vogal temática)? Explique cada um."},
        {"title": "Processos de Formação das Palavras", "range": (39, 42),
         "instruction": "Explique os processos de formação de palavras por derivação (prefixal, sufixal, parassintética, regressiva, imprópria) e composição (justaposição e aglutinação)."},
        {"title": "Morfologia - Substantivos: Classificação, Coletivos e Gênero", "range": (42, 49),
         "instruction": "Explique as classificações dos substantivos, os tipos de gênero (biformes, uniformes: epicenos, sobrecomuns e comuns de dois gêneros) e forneça exemplos."},
        {"title": "Flexão de Plural dos Substantivos Simples e Compostos", "range": (49, 56),
         "instruction": "Quais são as regras para a formação do plural dos substantivos compostos na Língua Portuguesa? Detalhe os casos em que variam ambos os elementos, apenas o primeiro, apenas o segundo ou nenhum."},
        {"title": "Morfologia - Numerais e suas Flexões", "range": (57, 60),
         "instruction": "Como se classificam os numerais (cardinais, ordinais, multiplicativos e fracionários) e quais são as regras de flexão e leitura em concursos?"},
        {"title": "Morfologia - Adjetivos: Classificação, Locuções e Flexão de Grau", "range": (60, 68),
         "instruction": "Explique a função do adjetivo, as locuções adjetivas correspondentes e as flexões de gênero, número e grau (comparativo e superlativo)."},
        {"title": "Sintaxe - Concordância Nominal: Regras Gerais e Casos Especiais", "range": (69, 74),
         "instruction": "Apresente as principais regras de concordância nominal, destacando expressões como 'é proibido / é proibida', 'bastante / bastantes', 'anexo / em anexo', 'meio / meia' e adjetivo anteposto/posposto a vários substantivos."},
        {"title": "Morfologia - Pronomes Pessoais, Retos, Oblíquos e de Tratamento", "range": (75, 83),
         "instruction": "Explique o emprego dos pronomes pessoais retos e oblíquos (o, a, os, as vs. lhe, lhes) e as regras para o uso adequado de pronomes de tratamento."},
        {"title": "Morfologia - Pronomes Relativos e Demonstrativos", "range": (83, 89),
         "instruction": "Explique como utilizar corretamente os pronomes demonstrativos (este, esse, aquele no espaço, no tempo e no texto) e o emprego do pronome relativo 'cujo'."},
        {"title": "Sintaxe - Colocação Pronominal (Próclise, Mesóclise e Ênclise)", "range": (90, 95),
         "instruction": "Quais são as palavras atrativas que exigem próclise e quais são as regras de mesóclise e ênclise, inclusive em locuções verbais?"},
        {"title": "Morfologia - Artigos e Preposições", "range": (96, 100),
         "instruction": "Explique o papel morfológico e semântico dos artigos e das preposições na coesão textual e na estruturação sintática."},
        {"title": "Morfologia - Verbos: Estrutura, Modos, Tempos e Formas Nominais", "range": (100, 110),
         "instruction": "Explique a estrutura do verbo, o valor semântico dos modos indicativo, subjuntivo e imperativo, e como se formam os tempos compostos."},
        {"title": "Morfologia - Verbos Notáveis, Irregulares e Defectivos", "range": (110, 116),
         "instruction": "Quais são as particularidades de conjugação dos verbos irregulares e defectivos mais cobrados em concursos (como pôr, haver, reaver, ver, vir e intervir)?"},
        {"title": "Vozes Verbais (Ativa, Passiva Analítica/Sintética e Reflexiva)", "range": (117, 120),
         "instruction": "Explique as vozes verbais e ensine o passo a passo da transposição da voz ativa para a voz passiva analítica e passiva sintética."},
        {"title": "Sintaxe - Predicação Verbal e Transitividade", "range": (120, 123),
         "instruction": "Como classificar os verbos quanto à transitividade (intransitivos, transitivos diretos, transitivos indiretos, transitivos diretos e indiretos e de ligação)?"},
        {"title": "Sintaxe - Regência Verbal e Verbos de Regência Oscilante", "range": (123, 131),
         "instruction": "Explique a regência dos principais verbos da Língua Portuguesa que mudam de sentido conforme a preposição (aspirar, assistir, visar, obedecer, preferir, esquecer/lembrar)."},
        {"title": "Sintaxe - Tipos de Sujeito", "range": (132, 138),
         "instruction": "Diferencie sujeito simples, composto, oculto/desinencial, indeterminado e oração sem sujeito (verbos impessoais como haver e fazer)."},
        {"title": "Sintaxe - Concordância Verbal: Regras Gerais e Sujeito Simples/Composto", "range": (138, 145),
         "instruction": "Apresente as regras de concordância verbal com sujeito simples, composto, expressões partitivas, porcentagens e núcleos ligados por 'ou' e 'com'."},
        {"title": "Sintaxe - Termos da Oração: Complementos Verbais, Agente da Passiva, Aposto e Vocativo", "range": (146, 150),
         "instruction": "Explique a função do objeto direto, objeto indireto, agente da passiva, aposto e vocativo, mostrando como identificá-los em uma oração."},
        {"title": "Sintaxe - Advérbios e Adjuntos Adverbiais", "range": (150, 160),
         "instruction": "Quais são as circunstâncias expressas pelos adjuntos adverbiais e como diferenciá-los dos predicativos e adjuntos adnominais?"},
        {"title": "Sintaxe - Distinção entre Adjunto Adnominal e Complemento Nominal", "range": (160, 162),
         "instruction": "Apresente critérios claros e objetivos para distinguir um Complemento Nominal de um Adjunto Adnominal quando preposicionados."},
        {"title": "Morfossintaxe - As Múltiplas Funções da Palavra QUE", "range": (162, 165),
         "instruction": "Explique as principais funções morfológicas e sintáticas da palavra 'que' (conjunção integrante, pronome relativo, advérbio, partícula expletiva, etc.)."},
        {"title": "Morfossintaxe - As Múltiplas Funções da Palavra SE", "range": (165, 171),
         "instruction": "Como identificar se a palavra 'se' atua como pronome apassivador (partícula apassivadora) ou índice de indeterminação do sujeito (IIS) e quais suas outras funções?"},
        {"title": "Sintaxe - Ocorrência, Proibição e Opcionalidade da Crase", "range": (171, 177),
         "instruction": "Explique detalhadamente quando o uso do acento grave indicativo de crase é obrigatório, proibido e facultativo segundo a norma culta."},
        {"title": "Problemas Gerais da Língua Culta e Emprego de Expressões", "range": (178, 181),
         "instruction": "Explique o uso correto de expressões como 'onde/aonde', 'mau/mal', 'há/a', 'a par/ao par', 'ao encontro de/de encontro a' e 'a fim de/afim'."},
        {"title": "Sintaxe do Período Composto - Orações Coordenadas e Subordinadas", "range": (181, 185),
         "instruction": "Explique a classificação das orações coordenadas e das orações subordinadas (substantivas, adjetivas e adverbiais) e suas conjunções típicas."},
        {"title": "Pontuação - Uso da Vírgula e Demais Sinais", "range": (185, 190),
         "instruction": "Quais são os casos obrigatórios e proibidos de uso da vírgula no período simples e no período composto?"},
        {"title": "Estilística - Figuras de Linguagem (Sintaxe, Palavras e Pensamento)", "range": (190, 196),
         "instruction": "Defina e exemplifique as principais figuras de linguagem: metáfora, metonímia, antítese, paradoxo, hipérbole, eufemismo, pleonasmo, zeugma e elipse."},
        {"title": "Dicas de Gramática e Macetes para Concursos", "range": (196, 200),
         "instruction": "Apresente dicas práticas e macetes gramaticais indispensáveis para resolução rápida de questões de português em provas de concurso público."},
        {"title": "Resumo de Fixação e Mnemônicos Finais", "range": (200, 214),
         "instruction": "Apresente um resumo esquemático de revisão com as regras mais recorrentes de ortografia, acentuação, crase e colocação pronominal para concursos."}
    ]

    items = []
    for idx, top in enumerate(theory_topics):
        p_start, p_end = top["range"]
        raw_content = ""
        for p in range(p_start, min(p_end + 1, len(doc))):
            raw_content += "\n" + clean_page(doc[p])

        # Remove eventuais gabaritos que caíram no final do capítulo
        m_resp = re.search(r'Respostas\b.*', raw_content, flags=re.IGNORECASE)
        if m_resp:
            raw_content = raw_content[:m_resp.start()]

        clean_text = raw_content.strip()
        if len(clean_text) < 100:
            continue

        # Formata o output pedagógico
        formatted_output = f"# {top['title']}\n\n{clean_text}"

        item = {
            "id": f"teoria_topico_{idx+1:02d}",
            "type": "teoria",
            "source": f"Apostila de Português para Concursos - {top['title']}",
            "topic": top["title"],
            "instruction": top["instruction"],
            "input": "",
            "output": formatted_output,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": top["instruction"]},
                {"role": "assistant", "content": formatted_output}
            ]
        }
        items.append(item)

    return items

def main():
    print(f"Abrindo arquivo {PDF_PATH}...")
    doc = pymupdf.open(PDF_PATH)
    print(f"Total de páginas: {len(doc)}")

    print("\n1. Extraindo as 20 baterias de questões de concurso (páginas 215 a 238)...")
    concurso_items = extract_concurso_batteries(doc)
    print(f"-> {len(concurso_items)} questões de concurso com gabarito extraídas!")

    print("\n2. Extraindo exercícios de fixação dos capítulos...")
    chap_items = extract_chapter_exercises(doc)
    print(f"-> {len(chap_items)} exercícios de capítulos com gabarito extraídos!")

    print("\n3. Extraindo unidades teóricas e conceituais estruturadas...")
    theory_items = extract_theory_units(doc)
    print(f"-> {len(theory_items)} tópicos teóricos aprofundados extraídos!")

    all_dataset = theory_items + chap_items + concurso_items
    print(f"\nTotal geral de registros no dataset: {len(all_dataset)}")

    print(f"Gravando arquivo JSONL em {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for entry in all_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("Gravação concluída com sucesso!")

    # Estatísticas
    type_counts = {}
    topic_counts = {}
    for it in all_dataset:
        t = it["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
        top = it["topic"]
        topic_counts[top] = topic_counts.get(top, 0) + 1

    stats = {
        "total_registros": len(all_dataset),
        "por_tipo": type_counts,
        "por_topico": topic_counts
    }

    with open("dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\nResumo das estatísticas gerado em dataset_stats.json!")

if __name__ == "__main__":
    main()
