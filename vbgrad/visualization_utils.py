from graphviz import Digraph


def backtrace_connections(root_node):
    visited_nodes, edges = set(), set()

    def find_connected_nodes(selected_node):
        if selected_node not in visited_nodes:
            visited_nodes.add(selected_node)
            for child in selected_node._children:
                edges.add((child, selected_node))
                find_connected_nodes(child)

    find_connected_nodes(root_node)
    return visited_nodes, edges


def draw_graph(root_node):
    graph = Digraph(format="svg", graph_attr={"rankdir": "LR"})
    nodes, edges = backtrace_connections(root_node)

    for node in nodes:
        uuid = str(id(node))
        graph.node(
            name=uuid,
            label=f"{node.variable_name} | Parameter {node.value:.4f}",
            shape="box",
        )

        # Here we create a node just to visualize the operation that took place
        if node._operation:
            graph.node(name=uuid + node._operation, label=node._operation, shape="oval")
            graph.edge(tail_name=uuid + node._operation, head_name=uuid)

    for node1, node2 in edges:
        uuid1 = str(id(node1))
        uuid2 = str(id(node2)) + node2._operation
        graph.edge(tail_name=uuid1, head_name=uuid2)

    return graph
