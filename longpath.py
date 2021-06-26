"""
给出一段 c/cpp 的 code，生成对应的 longpath 和 natural sequence (NS)。


使用 clang 参考：
https://stackoverflow.com/questions/26000876/how-to-solve-the-loading-error-of-clangs-python-binding

配置环境变量：
export DYLD_LIBRARY_PATH=/usr/local/Cellar/llvm/11.1.0/lib/

配置环境变量(PyCharm)：
https://stackoverflow.com/questions/42708389/how-to-set-environment-variables-in-pycharm

@Time    : 6/25/21
@Author  : Wenbo
"""
import networkx as nx
import os
import clang.cindex


# # This cell might not be needed for you.
# clang.cindex.Config.set_library_file(
#     '/usr/lib/llvm-7/lib/libclang-7.so.1'
# )


def generate_ast_roots(code):
    """
    Takes in a list of files/datapoints from juliet.csv.zip (as loaded with pandas) matching one particular
    testcase, and preprocesses it ready for the feature matrix.
    """
    index = clang.cindex.Index.create()
    parse_list = [('test.cpp', code)]
    translation_unit = index.parse(
        path='test.cpp',
        unsaved_files=parse_list,
    )
    ast_root = translation_unit.cursor

    concretise_ast(ast_root)
    number_ast_nodes(ast_root)

    return ast_root


def generate_features(ast_root):
    """
    Given a concretised & numbered clang ast, return a dictionary of
    features in the form:
        {
            <node_id>: [<degree>, <type>, <identifier>, <line_num>],
            ...
        }
    """
    features = {}

    def walk_tree_and_set_features(node):
        out_degree = len(node.children)
        in_degree = 1
        degree = out_degree + in_degree

        features[node.identifier] = [degree, str(node.kind), node.displayname, node.location.line]

        for child in node.children:
            walk_tree_and_set_features(child)

    walk_tree_and_set_features(ast_root)

    return features

def concretise_ast(node):
    """
    Everytime you run .get_children() on a clang ast node, it
    gives you new objects. So if you want to modify those objects
    they will lose their changes everytime you walk the tree again.
    To avoid this problem, concretise_ast walks the tree once,
    saving the resulting list from .get_children() into a a concrete
    list inside the .children.
    You can then use .children to consistently walk over tree, and
    it will give you the same objects each time.
    """
    node.children = list(node.get_children())

    for child in node.children:
        counter = concretise_ast(child)


def number_ast_nodes(node, counter=1):
    """
    Given a concretised clang ast, assign each node with a unique
    numerical identifier. This will be accessible via the .identifier
    attribute of each node.
    """
    node.identifier = counter
    counter += 1

    node.children = list(node.get_children())
    for child in node.children:
        counter = number_ast_nodes(child, counter)

    return counter

def generate_edgelist(ast_root):
    """
    Given a concretised & numbered clang ast, return a list of edges
    in the form:
        [
            [<start_node_id>, <end_node_id>],
            ...
        ]
    """
    edges = []

    def walk_tree_and_add_edges(node):
        for child in node.children:
            edges.append([node.identifier, child.identifier])
            walk_tree_and_add_edges(child)

    walk_tree_and_add_edges(ast_root)

    return edges

def generate_long_path(ast_root):
    """
    Given a concretised & numbered clang ast, return long paths
    """
    long_path = []
    edgelist = generate_edgelist(ast_root)
    G = nx.DiGraph()
    G.add_edges_from(edgelist)
    leafnodes = [x for x in G.nodes() if G.out_degree(x)==0 and G.in_degree(x)==1]
    rootnode = [x for x in G.nodes() if G.in_degree(x)==0]
    for node in leafnodes:
        long_path.append(nx.shortest_path(G, source=rootnode[0], target=node))
    return long_path


def generate_natural_sequence(code):
    res = []
    idx = clang.cindex.Index.create()
    tu = idx.parse('tmp.cpp', args=['-std=c++11'],
                   unsaved_files=[('tmp.cpp', code)], options=0)
    for t in tu.get_tokens(extent=tu.cursor.extent):
        print(t.kind, t.spelling, t.location)
        res.append(t.spelling)
    return res


if __name__ == '__main__':

    code = """
int who_am_i (void)
{
  struct passwd *pw;
  char *user = NULL;

  pw = getpwuid (geteuid ());
  if (pw)
    user = pw->pw_name;
  else if ((user = getenv ("USER")) == NULL)
    {
      fprintf (stderr, "I don't know!\n");
      return 1;
    }
  printf ("%s\n", user);
  
  fac(5);
  return 0;
}

int fac(int n) {
    return (n>1) ? n*fac(n-1) : 1;
}
"""
    ast_root = generate_ast_roots(code)
    print("=== ast:")
    print(ast_root)

    # get edge
    edge_list = generate_edgelist(ast_root)
    print("=== edge_list:")
    print(edge_list)

    # get nodes
    nodes = generate_features(ast_root)
    print("=== nodes:")
    print(nodes)

    # get long path
    long_path = generate_long_path(ast_root)
    print("=== long_path:")
    print(long_path)

    pass