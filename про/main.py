import os
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities

def run_pipeline(data_dir="data"):
    print("1. Загрузка данных...")
    edges_df = pd.read_parquet(os.path.join(data_dir, "edges.parquet"))
    nodes_df = pd.read_parquet(os.path.join(data_dir, "nodes.parquet"))
    
    print("2. Построение графа...")
    G = nx.from_pandas_edgedef(
        edges_df, 
        source='src', 
        target='dst', 
        edge_attr=['sum_kzt', 'n_tx', 'depth'], 
        create_using=nx.DiGraph()
    )
    
    # Убедимся, что все узлы из nodes_df присутствуют в графе
    for node in nodes_df['gid']:
        if not G.has_node(node):
            G.add_node(node)

    print("3. Расчет метрик для узлов...")
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    # Суммы входящих и исходящих транзакций по ребрам
    in_sum = edges_df.groupby('dst')['sum_kzt'].sum().to_dict()
    out_sum = edges_df.groupby('src')['sum_kzt'].sum().to_dict()
    
    # Кластеризация (сообщества) с помощью алгоритма Лувена (в неориентированном представлении)
    print("4. Кластеризация сети...")
    G_undirected = G.to_undirected()
    try:
        communities = list(louvain_communities(G_undirected, seed=42))
    except Exception:
        # Резервный вариант, если версия networkx требует другого вызова
        communities = list(nx.community.greedy_modularity_communities(G_undirected))
        
    node_to_cluster = {}
    cluster_data = []
    
    for cluster_id, comm in enumerate(communities):
        n_nodes = len(comm)
        seed_count = sum(1 for n in comm if nodes_df.loc[nodes_df['gid'] == n, 'is_seed'].any()) if 'is_seed' in nodes_df.columns else 0
        sum_kzt_int = edges_df[edges_df['src'].isin(comm) & edges_df['dst'].isin(comm)]['sum_kzt'].sum()
        top_gids = list(comm)[:5]
        
        cluster_data.append({
            "cluster_id": cluster_id,
            "n_nodes": n_nodes,
            "n_seed": seed_count,
            "sum_kzt_internal": sum_kzt_int,
            "top_gids": str(top_gids),
            "hypothesis": f"Кластер {cluster_id}: финансовая группа (узлов: {n_nodes}, seed: {seed_count})"
        })
        for node in comm:
            node_to_cluster[node] = cluster_id

    clusters_df = pd.DataFrame(cluster_data)

    print("5. Определение ролей и расчет priority_score...")
    results = []
    
    for node in G.nodes():
        in_deg = in_degrees.get(node, 0)
        out_deg = out_degrees.get(node, 0)
        inc_amt = in_sum.get(node, 0.0)
        out_amt = out_sum.get(node, 0.0)
        
        # Коэффициент пропуска (сколько ушло / сколько пришло)
        throughput_ratio = (out_amt / inc_amt) if inc_amt > 0 else 0.0
        
        # Логика определения ролей по эвристикам
        role = "peripheral"
        score = 0.5
        evidence = "Признаков специфической активности не обнаружено"
        
        # Проверяем артефакт обрыва 4-го колена (depth = 4 и out_deg == 0)
        node_row = nodes_df[nodes_df['gid'] == node]
        depth_val = node_row['press_depth'].values[0] if 'press_depth' in node_row.columns else (node_row['depth'].values[0] if 'depth' in node_row.columns else 0)
        
        if out_deg == 0 and depth_val >= 4:
            role = "terminal"
            score = 0.7
            evidence = "Конечный узел на границе обхода (depth=4, нет исходящих)"
        elif in_deg >= 5 and out_deg >= 5 and 0.8 <= throughput_ratio <= 1.2:
            role = "transit"
            score = 0.85
            evidence = f"Транзитный счет: входящих: {in_deg}, исходящих: {out_deg}, коэффициент пропуска: {throughput_ratio:.2f}"
        elif in_deg >= 5 and out_deg <= 2:
            role = "consolidator"
            score = 0.9
            evidence = f"Точка консолидации: аккумулирует средства от {in_deg} источников"
        elif out_deg >= 5 and in_deg <= 2:
            role = "distributor"
            score = 0.9
            evidence = f"Распределитель: веерная рассылка на {out_deg} получателей"
        elif in_deg == 0 and out_deg == 0:
            role = "peripheral"
            score = 0.3
            evidence = "Изолированный узел без активных транзакций в выборке"
        else:
            # Кандидаты в координаторы (высокая активность)
            if in_deg >= 3 and out_deg >= 3:
                role = "coordinator"
                score = 0.75
                evidence = f"Координатор: высокая сетевая активность (in: {in_deg}, out: {out_deg})"
                
        # Расчет приоритета для аналитика
        priority_score = min(1.0, (in_deg + out_deg) / 30.0 * score)
        
        results.append({
            "gid": node,
            "role": role,
            "role_score": score,
            "cluster_id": node_to_cluster.get(node, -1),
            "priority_score": priority_score,
            "evidence": evidence
        })

    nodes_roles_df = pd.DataFrame(results)
    
    # Формируем топ-20 узлов по приоритету
    top_nodes_df = nodes_roles_df.sort_values(by="priority_score", ascending=False).head(20).copy()
    top_nodes_df.reset_index(drop=True, inplace=True)
    top_nodes_df['rank'] = top_nodes_df.index + 1
    top_nodes_df['why'] = top_nodes_df['evidence']
    top_nodes_df = top_nodes_df[['rank', 'gid', 'role', 'priority_score', 'why']]

    print("6. Сохранение результатов в CSV...")
    nodes_roles_df.to_csv("nodes_roles.csv", index=False)
    clusters_df.to_csv("clusters.csv", index=False)
    top_nodes_df.to_csv("top_nodes.csv", index=False)
    print("Готово! Все 3 файла успешно созданы.")

if __name__ == "__main__":
    run_pipeline()