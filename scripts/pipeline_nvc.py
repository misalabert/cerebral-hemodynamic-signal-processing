import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# =====================================
# 1. CONFIGURAÇÃO
# =====================================
PASTA_DADOS = r"C:\Users\misal\OneDrive\Documentos\Projeto_Tese"

PASTA_SAIDA = os.path.join(PASTA_DADOS, "saida_pipeline_v8")
PASTA_FIGURAS_INDIVIDUAIS = os.path.join(PASTA_SAIDA, "figuras_individuais")
PASTA_FIGURAS_RESUMO = os.path.join(PASTA_SAIDA, "figuras_resumo")

os.makedirs(PASTA_SAIDA, exist_ok=True)
os.makedirs(PASTA_FIGURAS_INDIVIDUAIS, exist_ok=True)
os.makedirs(PASTA_FIGURAS_RESUMO, exist_ok=True)

# Convenção final:
# baseline = -20 a 0 s
# tarefa   = 0 a 60 s
BASELINE_WIN = (-20, 0)
ANALYSIS_WIN = (0, 60)
MIN_RESPONSE_WINDOW_S = 30  # mínimo aceitável para não excluir


# =====================================
# 2. FUNÇÕES AUXILIARES
# =====================================
def normalizar_nome_canal(nome):
    nome = str(nome).strip().lower()
    nome = re.sub(r"\s+", " ", nome)
    return nome


def normalizar_texto(x):
    return str(x).strip().lower()


def limpar_nome_arquivo(nome):
    nome = str(nome)
    nome = nome.replace(".mat", "")
    nome = re.sub(r"[^A-Za-z0-9_\-]+", "_", nome)
    return nome


def extrair_metadados(nome_arquivo):
    nome = nome_arquivo.lower()

    if "ctrl" in nome or "controle" in nome:
        grupo = "CTRL"
    elif "pd" in nome or "dp" in nome or "parkinson" in nome:
        grupo = "DP"
    else:
        grupo = "NA"

    if "supino" in nome:
        posicao = "supino"
    elif (
        "ort" in nome
        or "orto" in nome
        or "ortostat" in nome
        or "empe" in nome
        or "em_pe" in nome
        or "pe" in nome
        or "standing" in nome
        or "upright" in nome
    ):
        posicao = "ortostatica"
    else:
        posicao = "NA"

    if "ativo" in nome:
        tarefa = "ativo"
    elif "passivo" in nome:
        tarefa = "passivo"
    else:
        tarefa = "NA"

    return grupo, posicao, tarefa


def media_por_segundo(signal, time):
    duracao = int(np.floor(time[-1]))
    medias = []
    tempos = []

    for s in range(duracao + 1):
        idx = (time >= s) & (time < s + 1)
        if np.any(idx):
            medias.append(np.mean(signal[idx]))
            tempos.append(s)

    return np.array(tempos), np.array(medias)


def mean_in_window(df, col, t0, t1):
    x = df.loc[(df["t"] >= t0) & (df["t"] < t1), col].dropna()
    return float(x.mean()) if len(x) else np.nan


def pct_baseline(df, var):
    bl = mean_in_window(df, var, *BASELINE_WIN)
    return (df[var] / bl) * 100


def encontrar_canal_por_nome(titles, candidatos):
    titulos_norm = [normalizar_nome_canal(t) for t in titles]
    candidatos_norm = [normalizar_nome_canal(c) for c in candidatos]

    # igualdade exata
    for i, t in enumerate(titulos_norm):
        if t in candidatos_norm:
            return i

    # contém
    for i, t in enumerate(titulos_norm):
        for c in candidatos_norm:
            if c in t:
                return i

    return None


# =====================================
# 3. LEITURA ROBUSTA DO .MAT
# =====================================
def carregar_mat_labchart(caminho_arquivo):
    mat = loadmat(caminho_arquivo, squeeze_me=True, struct_as_record=False)

    # ===== FORMATO 1: simple =====
    if all(k in mat for k in ["data", "titles", "datastart", "dataend", "samplerate"]):
        return {
            "tipo": "simple",
            "mat": mat,
            "data": mat["data"],
            "titles": np.atleast_1d(mat["titles"]),
            "datastart": np.atleast_1d(mat["datastart"]),
            "dataend": np.atleast_1d(mat["dataend"]),
            "samplerate": np.atleast_1d(mat["samplerate"]),
        }

    # ===== FORMATO 2: block =====
    if all(k in mat for k in ["data_block1", "titles_block1", "ticktimes_block1"]):
        return {
            "tipo": "block",
            "mat": mat,
            "data_block": np.asarray(mat["data_block1"], dtype=float),
            "titles_block": np.atleast_1d(mat["titles_block1"]),
            "ticktimes_block": np.atleast_1d(mat["ticktimes_block1"]),
            "comtext_block1": mat.get("comtext_block1", None),
            "comtick_block1": mat.get("comtick_block1", None),
            "comchan_block1": mat.get("comchan_block1", None),
        }

    raise ValueError(
        "O arquivo não foi exportado em um formato reconhecido pelo pipeline. "
        f"Chaves encontradas: {sorted(mat.keys())}"
    )


def extrair_canais_labchart(caminho_arquivo):
    info = carregar_mat_labchart(caminho_arquivo)
    mat = info["mat"]

    # ========= FORMATO SIMPLE =========
    if info["tipo"] == "simple":
        data = info["data"]
        titles = info["titles"]
        datastart = info["datastart"]
        dataend = info["dataend"]
        samplerate = info["samplerate"]

        idx_mcav = encontrar_canal_por_nome(
            titles,
            ["MCAv", "MCAv E", "MCAV", "MCAV E", "MCAv 1", "MCAv1"]
        )
        idx_fc = encontrar_canal_por_nome(
            titles,
            ["FC", "Heart Rate", "HR"]
        )
        idx_map = encontrar_canal_por_nome(
            titles,
            ["MAP", "Mean Arterial", "Mean Arterial Pressure", "PAM"]
        )

        if idx_mcav is None:
            raise ValueError("Canal MCAv não encontrado pelos títulos.")
        if idx_fc is None:
            raise ValueError("Canal FC/HR não encontrado pelos títulos.")
        if idx_map is None:
            raise ValueError("Canal MAP/PAM não encontrado pelos títulos.")

        def pegar_canal_simple(i):
            inicio = int(datastart[i]) - 1
            fim = int(dataend[i])
            y = np.asarray(data[inicio:fim], dtype=float)
            fs = float(samplerate[i])
            nome = str(titles[i]).strip()
            return {"nome": nome, "y": y, "fs": fs}

        canais = {
            "mcav": pegar_canal_simple(idx_mcav),
            "fc": pegar_canal_simple(idx_fc),
            "map": pegar_canal_simple(idx_map),
        }

        return mat, canais, info

    # ========= FORMATO BLOCK =========
    elif info["tipo"] == "block":
        data_block = np.asarray(info["data_block"], dtype=float)
        titles_block = np.atleast_1d(info["titles_block"])
        ticktimes_block = np.asarray(info["ticktimes_block"], dtype=float).squeeze()
        t = np.ravel(ticktimes_block)

        if data_block.ndim == 1:
            data_block = data_block[:, None]

        # queremos tempo x canais
        if data_block.shape[0] == len(titles_block) and data_block.shape[1] == len(t):
            data_block = data_block.T
        elif data_block.shape[1] == len(titles_block) and data_block.shape[0] == len(t):
            pass
        elif data_block.shape[0] == len(titles_block):
            data_block = data_block.T
        elif data_block.shape[1] == len(titles_block):
            pass
        else:
            raise ValueError(
                f"Formato block inesperado: data_block shape={data_block.shape}, "
                f"n_titles={len(titles_block)}, n_ticks={len(t)}"
            )

        idx_mcav = encontrar_canal_por_nome(
            titles_block,
            ["MCAv", "MCAv E", "MCAV", "MCAV E", "MCAv 1", "MCAv1"]
        )
        idx_fc = encontrar_canal_por_nome(
            titles_block,
            ["FC", "Heart Rate", "HR"]
        )
        idx_map = encontrar_canal_por_nome(
            titles_block,
            ["MAP", "Mean Arterial", "Mean Arterial Pressure", "PAM"]
        )

        if idx_mcav is None:
            raise ValueError("Canal MCAv não encontrado pelos títulos (block).")
        if idx_fc is None:
            raise ValueError("Canal FC/HR não encontrado pelos títulos (block).")
        if idx_map is None:
            raise ValueError("Canal MAP/PAM não encontrado pelos títulos (block).")

        def pegar_canal_block(i):
            y = np.asarray(data_block[:, i], dtype=float)
            nome = str(titles_block[i]).strip()

            dt = np.diff(t)
            dt = dt[np.isfinite(dt)]
            dt = dt[dt > 0]
            if len(dt) == 0:
                raise ValueError(f"Não foi possível estimar a taxa de amostragem do canal {nome}.")

            fs = 1.0 / np.median(dt)

            return {"nome": nome, "y": y, "fs": fs, "t": t}

        canais = {
            "mcav": pegar_canal_block(idx_mcav),
            "fc": pegar_canal_block(idx_fc),
            "map": pegar_canal_block(idx_map),
        }

        return mat, canais, info


def diagnosticar_mat(caminho_arquivo):
    mat = loadmat(caminho_arquivo, squeeze_me=True, struct_as_record=False)

    print("Chaves do arquivo:")
    print(sorted(mat.keys()))

    if "titles" in mat:
        print("\nTítulos dos canais:")
        titles = np.atleast_1d(mat["titles"])
        for i, t in enumerate(titles):
            print(i, str(t))

    if "titles_block1" in mat:
        print("\nTítulos dos canais (block1):")
        titles = np.atleast_1d(mat["titles_block1"])
        for i, t in enumerate(titles):
            print(i, str(t))


# =====================================
# 4. COMENTÁRIOS / EVENTOS
# =====================================
def extrair_comentarios(mat):
    comentarios = []

    # formato clássico
    pares_possiveis = [
        ("comtime", "comtext"),
        ("commenttime", "commenttext"),
        ("eventtime", "eventtext"),
    ]

    for chave_tempo, chave_texto in pares_possiveis:
        if chave_tempo in mat and chave_texto in mat:
            tempos = np.atleast_1d(mat[chave_tempo])
            textos = np.atleast_1d(mat[chave_texto])

            n = min(len(tempos), len(textos))
            for i in range(n):
                try:
                    t = float(tempos[i])
                    txt = normalizar_texto(textos[i])
                    comentarios.append((t, txt))
                except Exception:
                    pass

    # formato block
    if "comtick_block1" in mat and "comtext_block1" in mat and "ticktimes_block1" in mat:
        comtick = np.atleast_1d(mat["comtick_block1"])
        comtext = np.atleast_1d(mat["comtext_block1"])
        ticktimes = np.atleast_1d(mat["ticktimes_block1"]).squeeze()
        ticktimes = np.ravel(ticktimes)

        n = min(len(comtick), len(comtext))
        for i in range(n):
            try:
                idx = int(comtick[i]) - 1
                if 0 <= idx < len(ticktimes):
                    t = float(ticktimes[idx])
                    txt = normalizar_texto(comtext[i])
                    comentarios.append((t, txt))
            except Exception:
                pass

    comentarios = sorted(comentarios, key=lambda x: x[0])
    return comentarios


def encontrar_janela_tarefa_por_comentario(mat, tarefa_esperada):
    comentarios = extrair_comentarios(mat)
    tarefa_esperada = normalizar_texto(tarefa_esperada)

    comentarios_tarefa = [(t, txt) for t, txt in comentarios if txt == tarefa_esperada]

    if len(comentarios_tarefa) >= 2:
        t_inicio_abs = comentarios_tarefa[0][0]
        t_fim_abs = comentarios_tarefa[1][0]
        origem = "comentarios"
        return t_inicio_abs, t_fim_abs, comentarios, origem

    elif len(comentarios_tarefa) == 1:
        t_inicio_abs = comentarios_tarefa[0][0]
        t_fim_abs = t_inicio_abs + 60.0
        origem = "comentario_unico"
        return t_inicio_abs, t_fim_abs, comentarios, origem

    else:
        # fallback compatível com a análise antiga
        t_inicio_abs = 20.0
        t_fim_abs = 80.0
        origem = "fallback_20_80"
        return t_inicio_abs, t_fim_abs, comentarios, origem


# =====================================
# 5. PROCESSAR UM ARQUIVO
# =====================================
def processar_arquivo_mat(caminho_arquivo, tarefa_esperada):
    mat, canais, info = extrair_canais_labchart(caminho_arquivo)

    mcav = canais["mcav"]["y"]
    hr   = canais["fc"]["y"]
    map_ = canais["map"]["y"]

    if info["tipo"] == "block":
        t_mcav_abs = np.asarray(canais["mcav"]["t"], dtype=float)
        t_hr_abs   = np.asarray(canais["fc"]["t"], dtype=float)
        t_map_abs  = np.asarray(canais["map"]["t"], dtype=float)
    else:
        t_mcav_abs = np.arange(len(mcav)) / canais["mcav"]["fs"]
        t_hr_abs   = np.arange(len(hr))   / canais["fc"]["fs"]
        t_map_abs  = np.arange(len(map_)) / canais["map"]["fs"]

    # reduzir para 1 Hz
    t_mcav_1hz_abs, mcav_1hz = media_por_segundo(mcav, t_mcav_abs)
    t_hr_1hz_abs, hr_1hz     = media_por_segundo(hr, t_hr_abs)
    t_map_1hz_abs, map_1hz   = media_por_segundo(map_, t_map_abs)

    # achar início/fim absoluto da tarefa
    t_inicio_abs, t_fim_abs, comentarios, origem_janela = encontrar_janela_tarefa_por_comentario(
        mat, tarefa_esperada
    )

    # converter tudo para tempo relativo à tarefa
    t_mcav_rel = t_mcav_1hz_abs - t_inicio_abs
    t_hr_rel   = t_hr_1hz_abs   - t_inicio_abs
    t_map_rel  = t_map_1hz_abs  - t_inicio_abs

    # alinhar FC e MAP ao eixo do MCAv
    hr_1hz_alinhado  = np.interp(t_mcav_rel, t_hr_rel, hr_1hz)
    map_1hz_alinhado = np.interp(t_mcav_rel, t_map_rel, map_1hz)

    n = min(len(t_mcav_rel), len(mcav_1hz), len(hr_1hz_alinhado), len(map_1hz_alinhado))

    df_1hz = pd.DataFrame({
        "t": t_mcav_rel[:n],
        "mcav": mcav_1hz[:n],
        "hr": hr_1hz_alinhado[:n],
        "map": map_1hz_alinhado[:n]
    })

    df_1hz["cvci"] = df_1hz["mcav"] / df_1hz["map"]

    # baseline
    mcav_bl = mean_in_window(df_1hz, "mcav", *BASELINE_WIN)
    map_bl  = mean_in_window(df_1hz, "map", *BASELINE_WIN)
    cvci_bl = mcav_bl / map_bl

    # janela de tarefa adaptativa
    t_max_disponivel = float(df_1hz["t"].max())
    t_fim_analise = min(ANALYSIS_WIN[1], t_max_disponivel)

    resposta = df_1hz[(df_1hz["t"] >= ANALYSIS_WIN[0]) & (df_1hz["t"] <= t_fim_analise)].copy()

    if resposta.empty or t_fim_analise < MIN_RESPONSE_WINDOW_S:
        raise ValueError("Janela de resposta insuficiente após alinhamento.")

    peak_mcav = float(resposta["mcav"].max())
    idx_peak = resposta["mcav"].idxmax()
    ttp = float(resposta.loc[idx_peak, "t"])

    delta_mcav_pct = ((peak_mcav - mcav_bl) / mcav_bl) * 100
    map_at_peak = float(resposta.loc[idx_peak, "map"])
    delta_map_pct = ((map_at_peak - map_bl) / map_bl) * 100
    cvci_peak = peak_mcav / map_at_peak
    delta_cvci_pct = ((cvci_peak - cvci_bl) / cvci_bl) * 100

    metricas = {
    "baseline_mcav": mcav_bl,
    "baseline_map": map_bl,
    "baseline_cvci": cvci_bl,
    "peak_mcav": peak_mcav,
    "ttp": ttp,
    "delta_mcav_%": delta_mcav_pct,
    "delta_map_%": delta_map_pct,
    "delta_cvci_%": delta_cvci_pct,
    "t_inicio_tarefa": 0.0,
    "t_fim_tarefa": t_fim_analise,
    "origem_janela": origem_janela,
    "canal_mcav": canais["mcav"]["nome"],
    "canal_fc": canais["fc"]["nome"],
    "canal_map": canais["map"]["nome"],
    "formato_mat": info["tipo"],
    "t_inicio_abs_real": t_inicio_abs,
    "t_fim_abs_real": t_fim_abs,
    "duracao_janela_utilizada_s": t_fim_analise,
}

    return df_1hz, metricas, comentarios


# =====================================
# 6. FIGURAS INDIVIDUAIS
# =====================================
def salvar_figura_paineis(df, nome_arquivo, t_fim_analise):
    nome_limpo = limpar_nome_arquivo(nome_arquivo)
    caminho_saida = os.path.join(PASTA_FIGURAS_INDIVIDUAIS, f"{nome_limpo}_paineis.png")

    mcav_pct = pct_baseline(df, "mcav")
    map_pct  = pct_baseline(df, "map")
    cvci_pct = pct_baseline(df, "cvci")

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axs[0].plot(df["t"], mcav_pct, linewidth=2)
    axs[0].axhline(100, linestyle="--")
    axs[0].axvline(0, linestyle="--")
    axs[0].axvline(t_fim_analise, linestyle="--")
    axs[0].set_ylabel("MCAv (% baseline)")
    axs[0].set_title(nome_arquivo)

    axs[1].plot(df["t"], map_pct, linewidth=2)
    axs[1].axhline(100, linestyle="--")
    axs[1].axvline(0, linestyle="--")
    axs[1].axvline(t_fim_analise, linestyle="--")
    axs[1].set_ylabel("MAP (% baseline)")

    axs[2].plot(df["t"], cvci_pct, linewidth=2)
    axs[2].axhline(100, linestyle="--")
    axs[2].axvline(0, linestyle="--")
    axs[2].axvline(t_fim_analise, linestyle="--")
    axs[2].set_ylabel("CVCi (% baseline)")
    axs[2].set_xlabel("Tempo (s)")

    plt.tight_layout()
    fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)


def salvar_figura_mcav_ttp(df, nome_arquivo, t_fim_analise):
    nome_limpo = limpar_nome_arquivo(nome_arquivo)
    caminho_saida = os.path.join(PASTA_FIGURAS_INDIVIDUAIS, f"{nome_limpo}_mcav_ttp.png")

    mcav_bl = mean_in_window(df, "mcav", *BASELINE_WIN)
    resposta = df[(df["t"] >= 0) & (df["t"] <= t_fim_analise)].copy()

    idx_peak = resposta["mcav"].idxmax()
    peak_mcav = float(resposta.loc[idx_peak, "mcav"])
    ttp = float(resposta.loc[idx_peak, "t"])

    fig = plt.figure(figsize=(9, 6))
    plt.axvspan(-20, 0, alpha=0.15, label="Baseline")
    plt.axvspan(0, t_fim_analise, alpha=0.10, label="Tarefa")
    plt.plot(df["t"], df["mcav"], linewidth=2.5, label="MCAv")
    plt.axhline(mcav_bl, linestyle="--", linewidth=1.2, label="Baseline MCAv")
    plt.axvline(0, linestyle="--", linewidth=1.2, label="Início da tarefa")
    plt.axvline(t_fim_analise, linestyle="--", linewidth=1.2, label="Fim da janela")
    plt.axvline(ttp, linestyle=":", linewidth=1.8, label="TTP")
    plt.scatter([ttp], [peak_mcav], s=80, zorder=5)
    plt.text(ttp + 1, peak_mcav, f"TTP = {ttp:.0f}s", va="bottom")

    plt.xlabel("Tempo (s)")
    plt.ylabel("MCAv")
    plt.title(nome_arquivo)
    plt.legend()
    plt.tight_layout()

    fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =====================================
# 7. LOOP AUTOMÁTICO
# =====================================
resultados = []
erros = []

for nome_arquivo in arquivos_mat:
    caminho = os.path.join(PASTA_DADOS, nome_arquivo)
    grupo, posicao, tarefa = extrair_metadados(nome_arquivo)

    linha = {
        "arquivo": nome_arquivo,
        "grupo": grupo,
        "posicao": posicao,
        "tarefa": tarefa,

        "baseline_mcav": np.nan,
        "baseline_map": np.nan,
        "baseline_cvci": np.nan,
        "peak_mcav": np.nan,
        "ttp": np.nan,
        "delta_mcav_%": np.nan,
        "delta_map_%": np.nan,
        "delta_cvci_%": np.nan,

        "t_inicio_tarefa": np.nan,
        "t_fim_tarefa": np.nan,
        "origem_janela": "",
        "canal_mcav": "",
        "canal_fc": "",
        "canal_map": "",
        "formato_mat": "",
        "t_inicio_abs_real": np.nan,
        "t_fim_abs_real": np.nan,
        "duracao_janela_utilizada_s": np.nan,

        "status": "ok",
        "erro": ""
    }

    try:
        df_1hz, metricas, comentarios = processar_arquivo_mat(caminho, tarefa)

        # preencher linha com as métricas calculadas
        linha.update(metricas)

        # salvar figuras individuais só se processou com sucesso
        try:
            t_fim_analise = metricas["duracao_janela_utilizada_s"]
            salvar_figura_paineis(df_1hz, nome_arquivo, t_fim_analise)
            salvar_figura_mcav_ttp(df_1hz, nome_arquivo, t_fim_analise)
        except Exception as e_fig:
            print(f"[AVISO FIGURA] {nome_arquivo}: {e_fig}")

    except Exception as e:
        linha["status"] = "erro"
        linha["erro"] = str(e)
        erros.append({
            "arquivo": nome_arquivo,
            "grupo": grupo,
            "posicao": posicao,
            "tarefa": tarefa,
            "erro": str(e)
        })
        print(f"[ERRO] {nome_arquivo}: {e}")

    resultados.append(linha)


# =====================================
# 8. TABELAS FINAIS
# =====================================
df_resultados = pd.DataFrame(resultados)
df_erros = pd.DataFrame(erros)

colunas_esperadas = [
    "arquivo", "grupo", "posicao", "tarefa",
    "baseline_mcav", "baseline_map", "baseline_cvci",
    "peak_mcav", "ttp",
    "delta_mcav_%", "delta_map_%", "delta_cvci_%",
    "t_inicio_tarefa", "t_fim_tarefa", "origem_janela",
    "canal_mcav", "canal_fc", "canal_map", "formato_mat",
    "t_inicio_abs_real", "t_fim_abs_real", "duracao_janela_utilizada_s",
    "status", "erro"
]

for c in colunas_esperadas:
    if c not in df_resultados.columns:
        df_resultados[c] = np.nan

df_resultados = df_resultados[colunas_esperadas]

print("\nTabela final:")
print(df_resultados.head())

if not df_erros.empty:
    print("\nArquivos com erro:")
    print(df_erros)


# =====================================
# 9. SALVAR PLANILHAS
# =====================================
saida_resultados = os.path.join(PASTA_SAIDA, "resultados_pipeline_v8.xlsx")
df_resultados.to_excel(saida_resultados, index=False)

if not df_erros.empty:
    saida_erros = os.path.join(PASTA_SAIDA, "erros_pipeline_v8.xlsx")
    df_erros.to_excel(saida_erros, index=False)


# =====================================
# 10. RESUMOS AUTOMÁTICOS
# =====================================
df_validos = df_resultados[df_resultados["status"] == "ok"].copy()

if not df_validos.empty:
    resumo_grupo = (
        df_validos.groupby("grupo")[["ttp", "delta_mcav_%", "delta_map_%", "delta_cvci_%"]]
        .agg(["mean", "std", "count"])
        .round(2)
    )

    resumo_grupo_tarefa = (
        df_validos.groupby(["grupo", "tarefa"])[["ttp", "delta_mcav_%", "delta_map_%", "delta_cvci_%"]]
        .agg(["mean", "std", "count"])
        .round(2)
    )

    resumo_grupo_posicao_tarefa = (
        df_validos.groupby(["grupo", "posicao", "tarefa"])[["ttp", "delta_mcav_%", "delta_map_%", "delta_cvci_%"]]
        .agg(["mean", "std", "count"])
        .round(2)
    )

    resumo_grupo.to_excel(os.path.join(PASTA_SAIDA, "resumo_por_grupo.xlsx"))
    resumo_grupo_tarefa.to_excel(os.path.join(PASTA_SAIDA, "resumo_por_grupo_tarefa.xlsx"))
    resumo_grupo_posicao_tarefa.to_excel(os.path.join(PASTA_SAIDA, "resumo_por_grupo_posicao_tarefa.xlsx"))

    print("\nResumo por grupo:")
    print(resumo_grupo)

    print("\nResumo por grupo e tarefa:")
    print(resumo_grupo_tarefa)

    print("\nResumo por grupo, posição e tarefa:")
    print(resumo_grupo_posicao_tarefa)


# =====================================
# 11. FIGURAS-RESUMO
# =====================================
if not df_validos.empty:
    df_plot = df_validos.copy()

    ordem_grupos = ["CTRL", "DP"]
    df_plot["grupo"] = pd.Categorical(df_plot["grupo"], categories=ordem_grupos, ordered=True)

    # figura 1: TTP por grupo
    fig = plt.figure(figsize=(6, 5))
    for i, grp in enumerate(ordem_grupos):
        vals = df_plot[df_plot["grupo"] == grp]["ttp"].dropna().values
        if len(vals) == 0:
            continue
        media = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        plt.bar(i, media, yerr=sem, capsize=5)
        jitter = np.random.normal(i, 0.04, size=len(vals))
        plt.scatter(jitter, vals, s=40, zorder=3)

    plt.xticks([0, 1], ordem_grupos)
    plt.ylabel("TTP (s)")
    plt.title("Tempo para pico por grupo")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIGURAS_RESUMO, "ttp_por_grupo.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # figura 2: ΔMCAv por grupo
    fig = plt.figure(figsize=(6, 5))
    for i, grp in enumerate(ordem_grupos):
        vals = df_plot[df_plot["grupo"] == grp]["delta_mcav_%"].dropna().values
        if len(vals) == 0:
            continue
        media = vals.mean()
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        plt.bar(i, media, yerr=sem, capsize=5)
        jitter = np.random.normal(i, 0.04, size=len(vals))
        plt.scatter(jitter, vals, s=40, zorder=3)

    plt.xticks([0, 1], ordem_grupos)
    plt.ylabel("ΔMCAv (%)")
    plt.title("ΔMCAv por grupo")
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIGURAS_RESUMO, "delta_mcav_por_grupo.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # figura 3: TTP por grupo e tarefa
    plot_df = df_plot[df_plot["tarefa"].isin(["ativo", "passivo"])].copy()
    grupos = ["CTRL", "DP"]
    tarefas = ["ativo", "passivo"]
    x_positions = np.arange(len(tarefas))
    bar_width = 0.35

    fig = plt.figure(figsize=(7, 5))
    for j, grp in enumerate(grupos):
        means = []
        sems = []
        for tarefa in tarefas:
            vals = plot_df[(plot_df["grupo"] == grp) & (plot_df["tarefa"] == tarefa)]["ttp"].dropna().values
            means.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

        offset = (j - 0.5) * bar_width
        plt.bar(x_positions + offset, means, width=bar_width, yerr=sems, capsize=4, label=grp)

        for i, tarefa in enumerate(tarefas):
            vals = plot_df[(plot_df["grupo"] == grp) & (plot_df["tarefa"] == tarefa)]["ttp"].dropna().values
            jitter = np.random.normal(x_positions[i] + offset, 0.03, size=len(vals))
            plt.scatter(jitter, vals, s=35, zorder=3)

    plt.xticks(x_positions, ["Ativo", "Passivo"])
    plt.ylabel("TTP (s)")
    plt.title("TTP por tarefa e grupo")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PASTA_FIGURAS_RESUMO, "ttp_por_tarefa_e_grupo.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # figura 4: painel estilo paper
    variaveis = [
        ("delta_mcav_%", "ΔMCAv (%)", "A"),
        ("delta_map_%", "ΔMAP (%)", "B"),
        ("delta_cvci_%", "ΔCVCi (%)", "C"),
        ("ttp", "TTP (s)", "D")
    ]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()

    for ax, (var, ylabel, letra) in zip(axs, variaveis):
        grupos_series = df_plot.groupby("grupo", observed=True)[var]
        media = grupos_series.mean().reindex(ordem_grupos)
        sem = grupos_series.sem().reindex(ordem_grupos)

        x = np.arange(len(ordem_grupos))
        ax.bar(x, media, yerr=sem, capsize=5, edgecolor="black", linewidth=1.2)

        for i, grupo in enumerate(ordem_grupos):
            valores = df_plot[df_plot["grupo"] == grupo][var].dropna().values
            x_jitter = np.random.normal(i, 0.05, size=len(valores))
            ax.scatter(x_jitter, valores, s=45, edgecolors="black", linewidths=0.8, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(ordem_grupos)
        ax.set_ylabel(ylabel)
        ax.set_title(letra, loc="left", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Resposta hemodinâmica cerebral", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PASTA_FIGURAS_RESUMO, "painel_estilo_paper.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# =====================================
# 12. MENSAGENS FINAIS
# =====================================
print("\nResultados salvos em:")
print(saida_resultados)

if not df_erros.empty:
    print("\nErros salvos em:")
    print(saida_erros)

print("\nFiguras individuais salvas em:")
print(PASTA_FIGURAS_INDIVIDUAIS)

print("\nFiguras-resumo salvas em:")
print(PASTA_FIGURAS_RESUMO)

print("\nPipeline v8 finalizado.")
