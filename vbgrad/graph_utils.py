def extract_topological_map(root_node):
    topological_map = []
    visited_nodes = set()

    def traverse_nodes(node):
        if node not in visited_nodes:
            visited_nodes.add(node)
            for child in node._children:
                traverse_nodes(child)
            topological_map.append(node)

    traverse_nodes(root_node)

    return list(reversed(topological_map))
