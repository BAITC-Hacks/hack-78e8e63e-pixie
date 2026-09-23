import os
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.community import louvain_communities


def run_pipeline(data_dir="data", out_dir="."):
    print("1. Загрузка данных...")
    edges_df = pd.read_parquet(os.path.join(data_dir, "edges.parquet"))
    nodes_df = pd.read_parquet(os.path.join(data_dir, "nodes.parquet"))

    print("2. Построение графа...")
    G = nx.from_pandas_edgelist(
        edges_df,
        source='src',
        target='dst',
        edge_attr=['sum_kzt', 'n_tx', 'depth'],
        create_using=nx.DiGraph()
    )
    for node in nodes_df['gid']:
        if not G.has_node(node):
            G.add_node(node)

    print("3. Расчет метрик для узлов...")
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    in_sum = edges_df.groupby('dst')['sum_kzt'].sum().to_dict()
    out_sum = edges_df.groupby('src')['sum_kzt'].sum().to_dict()
    depth_map = nodes_df.set_index('gid')['depth'].to_dict()
    seed_map = nodes_df.set_index('gid')['is_seed'].to_dict() if 'is_seed' in nodes_df.columns else {}

    hubs, auth = nx.hits(G, max_iter=500)
    betweenness = nx.betweenness_centrality(G, normalized=True)
    pagerank = nx.pagerank(G, weight="sum_kzt")

    cycle_nodes = set()
    try:
        for cyc in nx.simple_cycles(G, length_bound=4):
            cycle_nodes.update(cyc)
    except Exception:
        pass

    print("4. Кластеризация сети...")
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = d.get("sum_kzt", 1.0)
        if UG.has_edge(u, v):
            UG[u][v]["weight"] += w
        else:
            UG.add_edge(u, v, weight=w)
    try:
        communities = list(louvain_communities(UG, weight="weight", seed=42))
    except Exception:
        communities = list(nx.community.greedy_modularity_communities(UG))

    node_to_cluster = {}
    role_by_node = {}
    cluster_data = []
    for cluster_id, comm in enumerate(communities):
        for node in comm:
            node_to_cluster[node] = cluster_id

    print("5. Определение ролей и расчет priority_score...")

    T_CONSOLIDATOR_IN = 5
    T_DISTRIBUTOR_OUT = 5
    T_LOWDEG = 2
    T_TRANSIT_DEG = 5
    PASS_LOW, PASS_HIGH = 0.6, 1.6

    mixed_nodes = [n for n in G.nodes() if in_degrees.get(n, 0) >= 3 and out_degrees.get(n, 0) >= 3]
    bt_values = sorted(betweenness.get(n, 0.0) for n in mixed_nodes)
    bt_threshold = bt_values[int(0.75 * (len(bt_values) - 1))] if bt_values else 1.0

    results = []
    for node in G.nodes():
        in_deg = in_degrees.get(node, 0)
        out_deg = out_degrees.get(node, 0)
        inc_amt = in_sum.get(node, 0.0)
        out_amt = out_sum.get(node, 0.0)
        is_seed = bool(seed_map.get(node, False))
        pt = None if (is_seed or inc_amt <= 0) else out_amt / inc_amt

        depth_val = depth_map.get(node, 0)
        truncated = (out_deg == 0 and depth_val >= 4)
        in_cycle = node in cycle_nodes
        cycle_note = " Входит в короткий возвратный цикл переводов (≤4 шага)." if in_cycle else ""

        if truncated:
            if in_deg >= T_CONSOLIDATOR_IN:
                role, score = "consolidator", 0.55
                evidence = (f"Вероятная консолидация, обрезанная границей обхода: вход от {in_deg} "
                            f"плательщиков (уверенность снижена — depth=4)")
            else:
                role, score = "terminal", 0.5
                evidence = f"Конечный узел на границе обхода depth=4, вход от {in_deg} — неотличим от истинного terminal"

        elif in_deg >= T_TRANSIT_DEG and out_deg >= T_TRANSIT_DEG and pt is not None and PASS_LOW <= pt <= PASS_HIGH:
            role, score = "transit", 0.85
            evidence = f"Признаки транзита: {in_deg} плательщиков → {out_deg} получателей, pass_through={pt:.2f}.{cycle_note}"

        elif in_deg >= T_CONSOLIDATOR_IN and (out_deg <= T_LOWDEG or in_deg >= 3 * max(out_deg, 1)):
            role = "consolidator"
            score = 0.9 if out_deg <= T_LOWDEG else 0.75
            evidence = (f"Признаки консолидации: получает от {in_deg} плательщиков, отдаёт {out_deg} "
                        f"(вход в {in_deg / max(out_deg, 1):.1f}× больше выхода).{cycle_note}")

        elif out_deg >= T_DISTRIBUTOR_OUT and (in_deg <= T_LOWDEG or out_deg >= 3 * max(in_deg, 1)):
            role = "distributor"
            score = 0.9 if in_deg <= T_LOWDEG else 0.75
            evidence = (f"Признаки распределения: {out_deg} получателей против {in_deg} входящих "
                        f"(выход в {out_deg / max(in_deg, 1):.1f}× больше входа).{cycle_note}")

        elif in_deg == 0 and out_deg == 0:
            role, score = "peripheral", 0.3
            evidence = "Нет ни одной связи в выборке (изолированный узел)"

        elif in_deg >= 3 and out_deg >= 3:
            if betweenness.get(node, 0.0) >= bt_threshold or in_cycle:
                role = "coordinator"
                score = 0.75 if in_cycle else 0.65
                evidence = (f"Признаки координации: {in_deg} вх./{out_deg} исх., betweenness={betweenness.get(node, 0):.3f} "
                            f"(топ по посредничеству среди узлов со смешанной активностью).{cycle_note}")
            else:
                role, score = "peripheral", 0.45
                evidence = (f"Смешанная активность без выраженной посреднической роли: "
                            f"{in_deg} вх./{out_deg} исх., betweenness={betweenness.get(node, 0):.3f}")

        else:
            role, score = "peripheral", 0.4
            evidence = f"Низкая активность: {in_deg} входящих, {out_deg} исходящих связей.{cycle_note}"

        evidence = evidence[:200]
        role_by_node[node] = role

        role_weight = {"consolidator": 1.0, "coordinator": 0.95, "distributor": 0.8,
                       "transit": 0.7, "terminal": 0.4, "peripheral": 0.1}[role]
        results.append({
            "gid": node,
            "role": role,
            "role_score": score,
            "cluster_id": node_to_cluster.get(node, -1),
            "_role_weight": role_weight,
            "_pagerank": pagerank.get(node, 0.0),
            "_deg": in_deg + out_deg,
            "_betweenness": betweenness.get(node, 0.0),
            "evidence": evidence,
            "_in_deg": in_deg,
            "_out_deg": out_deg,
            "_in_amt": inc_amt,
            "_out_amt": out_amt,
            "_in_cycle": in_cycle,
            "_truncated": truncated,
        })

    nodes_roles_df = pd.DataFrame(results)

    pr_max = nodes_roles_df["_pagerank"].max() or 1
    deg_max = nodes_roles_df["_deg"].max() or 1
    bt_max = nodes_roles_df["_betweenness"].max() or 1
    nodes_roles_df["priority_score"] = (
        0.4 * nodes_roles_df["_role_weight"]
        + 0.3 * (nodes_roles_df["_pagerank"] / pr_max)
        + 0.2 * (nodes_roles_df["_deg"] / deg_max)
        + 0.1 * (nodes_roles_df["_betweenness"] / bt_max)
    ).clip(0, 1)

    # это нужно и для SAR/resilience ниже, поэтому сохраняем отдельно перед сбросом
    nodes_roles_full_df = nodes_roles_df.copy()
    nodes_roles_df = nodes_roles_df.drop(columns=[
        "_role_weight", "_pagerank", "_deg", "_betweenness",
        "_in_deg", "_out_deg", "_in_amt", "_out_amt", "_in_cycle", "_truncated"
    ])

    for cluster_id, comm in enumerate(communities):
        n_nodes = len(comm)
        n_seed = sum(1 for n in comm if seed_map.get(n, False))
        sum_kzt_int = edges_df[edges_df['src'].isin(comm) & edges_df['dst'].isin(comm)]['sum_kzt'].sum()
        top_gids = sorted(comm, key=lambda n: pagerank.get(n, 0.0), reverse=True)[:5]
        role_counts = pd.Series([role_by_node.get(n) for n in comm]).value_counts()
        dominant_role = role_counts.index[0] if len(role_counts) else "peripheral"
        cluster_data.append({
            "cluster_id": cluster_id,
            "n_nodes": n_nodes,
            "n_seed": n_seed,
            "sum_kzt_internal": sum_kzt_int,
            "top_gids": str(top_gids),
            "hypothesis": (f"{n_nodes} узлов, {n_seed} seed, оборот внутри кластера {sum_kzt_int:,.0f} KZT, "
                           f"преобладающая роль: {dominant_role} ({role_counts.get(dominant_role, 0)} узлов)")
        })
    clusters_df = pd.DataFrame(cluster_data)

    top_nodes_df = nodes_roles_df.sort_values(by="priority_score", ascending=False).head(30).copy()
    top_nodes_df.reset_index(drop=True, inplace=True)
    top_nodes_df['rank'] = top_nodes_df.index + 1
    top_nodes_df['why'] = top_nodes_df['evidence']
    top_nodes_df = top_nodes_df[['rank', 'gid', 'role', 'priority_score', 'why']]

    print("6. Сохранение обязательных выгрузок в CSV...")
    os.makedirs(out_dir, exist_ok=True)
    nodes_roles_df.to_csv(os.path.join(out_dir, "nodes_roles.csv"), index=False)
    clusters_df.to_csv(os.path.join(out_dir, "clusters.csv"), index=False)
    top_nodes_df.to_csv(os.path.join(out_dir, "top_nodes.csv"), index=False)

    # === дополнительная выгрузка для экрана просмотра: рёбра с ролями узлов ===
    print("7. Экспорт рёбер для визуализации...")
    role_map = nodes_roles_df.set_index("gid")["role"].to_dict()
    cluster_map = nodes_roles_df.set_index("gid")["cluster_id"].to_dict()
    edges_export = edges_df[['src', 'dst', 'sum_kzt', 'n_tx', 'depth']].copy()
    edges_export['src_role'] = edges_export['src'].map(role_map)
    edges_export['dst_role'] = edges_export['dst'].map(role_map)
    edges_export['in_cycle'] = edges_export.apply(
        lambda r: (r['src'] in cycle_nodes and r['dst'] in cycle_nodes), axis=1)
    edges_export.to_csv(os.path.join(out_dir, "edges_export.csv"), index=False)

    # === опция 1: авто-генерация SAR-нарратива (шаблонная, детерминированная — explainable) ===
    print("8. Генерация SAR-нарративов для топ-узлов...")
    sar_narratives = generate_sar_narratives(nodes_roles_full_df, cluster_data, top_nodes_df)
    import json
    with open(os.path.join(out_dir, "sar_narratives.json"), "w", encoding="utf-8") as f:
        json.dump(sar_narratives, f, ensure_ascii=False, indent=2)

    # === опция 2: Key Player Problem — устойчивость сети к удалению топ-N узлов ===
    print("9. Анализ устойчивости сети (Key Player disruption)...")
    resilience_df = analyze_network_resilience(G, nodes_roles_full_df)
    resilience_df.to_csv(os.path.join(out_dir, "network_resilience.csv"), index=False)

    print(f"Готово! Файлы созданы в {out_dir}/")
    print(nodes_roles_df["role"].value_counts().to_string())
    return nodes_roles_df, clusters_df, top_nodes_df


def generate_sar_narratives(nodes_roles_full_df, cluster_data, top_nodes_df, max_nodes=30):
    """
    Автогенерация SAR-style (Suspicious Activity Report) нарратива по топ-узлам.
    Намеренно ШАБЛОННАЯ (не LLM) — детерминированно, воспроизводимо, без риска
    галлюцинаций и без "чёрного ящика" (см. ограничение ТЗ п.9: роль без
    объяснимого правила не засчитывается). Формулировки — как гипотезы,
    а не утверждения о виновности (требование ТЗ п.9, "Осторожность формулировок").
    """
    ROLE_LABELS_RU = {
        "consolidator": "точка консолидации средств",
        "distributor": "узел веерного распределения средств",
        "transit": "транзитный счёт",
        "coordinator": "узел с признаками координации сети",
        "terminal": "конечный получатель (на границе обхода)",
        "peripheral": "периферийный узел",
    }
    cluster_by_id = {c["cluster_id"]: c for c in cluster_data}
    idx = nodes_roles_full_df.set_index("gid")

    narratives = []
    for _, row in top_nodes_df.head(max_nodes).iterrows():
        gid = row["gid"]
        node_row = idx.loc[gid]
        role = row["role"]
        cl = cluster_by_id.get(int(node_row["cluster_id"]), {})
        flags = []
        if bool(node_row.get("_in_cycle")):
            flags.append("участвует в коротком возвратном цикле переводов (≤4 шага)")
        if bool(node_row.get("_truncated")):
            flags.append("находится на границе обхода (depth=4) — возможен неучтённый исходящий поток")

        narrative = (
            f"Клиент gid={gid}. Роль (гипотеза): {ROLE_LABELS_RU.get(role, role)} "
            f"(уверенность {node_row['role_score']:.2f}, приоритет {row['priority_score']:.2f}). "
            f"Входящих переводов: {int(node_row['_in_deg'])} на сумму {node_row['_in_amt']:,.0f} KZT; "
            f"исходящих: {int(node_row['_out_deg'])} на сумму {node_row['_out_amt']:,.0f} KZT. "
            f"Кластер #{int(node_row['cluster_id'])}: {cl.get('hypothesis', 'нет данных')}."
        )
        if flags:
            narrative += " Особые признаки: " + "; ".join(flags) + "."
        narrative += (
            " Рекомендация: включить в перечень на углублённую проверку и запрос "
            "дополнительной информации в правоохранительные органы. Вывод носит "
            "гипотетический характер и требует подтверждения аналитиком."
        )

        narratives.append({
            "gid": int(gid) if hasattr(gid, "item") else gid,
            "role": role,
            "priority_score": round(float(row["priority_score"]), 3),
            "narrative": narrative,
        })
    return narratives


def analyze_network_resilience(G, nodes_roles_full_df, top_k_values=(5, 10, 20, 30, 50, 81)):
    """
    Key Player Problem-style disruption-анализ (методика восходит к S. Borgatti,
    "Identifying sets of key players in a social network", 2006 — применяется
    NIJ/военной разведкой США для планирования disruption криминальных и
    террористических сетей). Идея: не просто "у кого выше centrality", а что
    РЕАЛЬНО произойдёт со связностью сети, если этих узлов не станет —
    распадается ли она на изолированные фрагменты (что и есть цель
    правоохранителей — не поймать одного курьера, а разрушить инфраструктуру).

    Метрика на каждом шаге: размер наибольшей компоненты связности (в %% от
    исходной) и число отдельных компонент после удаления top-K узлов по
    priority_score.
    """
    UG_base = G.to_undirected()
    total_nodes = UG_base.number_of_nodes()
    ordered = nodes_roles_full_df.sort_values("priority_score", ascending=False)["gid"].tolist()

    rows = []
    # baseline
    largest_cc0 = max((len(c) for c in nx.connected_components(UG_base)), default=0)
    rows.append({
        "removed_top_k": 0,
        "removed_gids": "",
        "largest_component_size": largest_cc0,
        "largest_component_pct": round(100 * largest_cc0 / total_nodes, 1) if total_nodes else 0,
        "n_components": nx.number_connected_components(UG_base),
    })

    for k in top_k_values:
        remove_set = ordered[:k]
        H = UG_base.copy()
        H.remove_nodes_from(remove_set)
        if H.number_of_nodes() == 0:
            largest_cc = 0
            n_comp = 0
        else:
            largest_cc = max((len(c) for c in nx.connected_components(H)), default=0)
            n_comp = nx.number_connected_components(H)
        rows.append({
            "removed_top_k": k,
            "removed_gids": str(remove_set),
            "largest_component_size": largest_cc,
            "largest_component_pct": round(100 * largest_cc / total_nodes, 1) if total_nodes else 0,
            "n_components": n_comp,
        })

    return pd.DataFrame(rows)


def launch_dashboard(out_dir=".", port=8765):
    """
    Поднимает локальный http-сервер поверх out_dir и открывает graph_viewer.html
    в браузере — так дашборд подхватывает CSV/JSON через fetch() автоматически,
    без ручного выбора файлов (как `streamlit run`, только без streamlit).
    """
    import http.server
    import socketserver
    import webbrowser
    import threading
    import functools
    import shutil

    viewer_name = "graph_viewer.html"
    dest = os.path.join(out_dir, viewer_name)
    if not os.path.exists(dest):
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), viewer_name)
        if os.path.exists(here):
            shutil.copy(here, dest)
        else:
            print(f"\nНе найден {viewer_name} — положи его рядом с main.py или в {out_dir}/, "
                  f"и запусти ещё раз, чтобы открылся дашборд.")
            return

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=out_dir)
    httpd = socketserver.TCPServer(("", port), handler)
    url = f"http://localhost:{port}/{viewer_name}"
    print(f"\nДэшборд: {url}")
    print("Ctrl+C — остановить сервер.")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")


if __name__ == "__main__":
    run_pipeline()
    launch_dashboard(out_dir=".")