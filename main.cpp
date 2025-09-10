// Arbor Porphyriana modeled with a Van Emde Boas Tree (VEB)
// + Dijkstra (unit weights) with timing + ASCII & Graphviz diagrams (ASCII-only).
//
// Build & run (example):
//   g++ -std=c++17 -O2 -o arbor main.cpp && ./arbor
//
// Diagram (Graphviz):
//   dot -Tpng porphyry.dot -o porphyry.png
//
// What this program does:
// 1) Implements a Van Emde Boas (VEB) tree to index all concept IDs.
// 2) Builds a Porphyrian-style taxonomy (sample "animal -> feline/canine -> cat... dog...",
//    plus a generator for an N-level synthetic tree).
// 3) Measures and prints build time and Dijkstra time (shortest path between terms).
// 4) Prints a compact textual view of the VEB clusters with their labels.
// 5) Renders the taxonomy as:
//    - ASCII tree in the console (ASCII characters only for portability).
//    - Graphviz DOT file (porphyry.dot) for a clean diagram.
//

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <algorithm>
using namespace std;
using namespace std::chrono;

// ----------------------------- Van Emde Boas Tree -----------------------------
class Van_Emde_Boas {
public:
    int universe_size;  // U
    int minimum;        // min key or -1 if empty
    int maximum;        // max key or -1 if empty
    Van_Emde_Boas* summary;                 // VEB(sqrt(U))
    vector<Van_Emde_Boas*> clusters;        // sqrt(U) clusters, each VEB(sqrt(U))

    explicit Van_Emde_Boas(int size): universe_size(size), minimum(-1), maximum(-1), summary(nullptr) {
        if (size <= 2) {
            clusters = vector<Van_Emde_Boas*>(0, nullptr);
        } else {
            int no_clusters = (int)ceil(sqrt((double)size));
            summary = new Van_Emde_Boas(no_clusters);
            clusters = vector<Van_Emde_Boas*>(no_clusters, nullptr);
            for (int i = 0; i < no_clusters; i++) {
                clusters[i] = new Van_Emde_Boas((int)ceil(sqrt((double)size)));
            }
        }
    }

    ~Van_Emde_Boas() {
        if (summary) delete summary;
        for (auto* c : clusters) delete c;
    }

    inline int high(int x) const { int d = (int)ceil(sqrt((double)universe_size)); return x / d; }
    inline int low(int x)  const { int m = (int)ceil(sqrt((double)universe_size)); return x % m; }
    inline int generate_index(int x, int y) const { int ru = (int)ceil(sqrt((double)universe_size)); return x * ru + y; }

    inline bool empty() const { return minimum == -1; }

    void empty_insert(int x) { minimum = maximum = x; }

    bool contains(int x) const {
        if (x == minimum || x == maximum) return true;
        if (universe_size <= 2) return false;
        int h = high(x), l = low(x);
        if (h < 0 || h >= (int)clusters.size()) return false;
        if (!clusters[h] || clusters[h]->empty()) return false;
        return clusters[h]->contains(l);
    }

    void insert(int x) {
        if (minimum == -1) { // empty tree
            empty_insert(x);
            return;
        }
        if (x < minimum) std::swap(x, minimum);

        if (universe_size > 2) {
            int h = high(x);
            int l = low(x);
            if (clusters[h]->minimum == -1) {
                summary->insert(h);
                clusters[h]->empty_insert(l);
            } else {
                clusters[h]->insert(l);
            }
        }
        if (x > maximum) maximum = x;
    }

    // Enumerate all keys stored in this VEB (demo only).
    void enumerate(vector<int>& out) const {
        if (minimum == -1) return;
        out.push_back(minimum);
        if (universe_size <= 2) {
            if (maximum != -1 && maximum != minimum) out.push_back(maximum);
            return;
        }
        int ru = (int)ceil(sqrt((double)universe_size));
        for (int h = 0; h < (int)clusters.size(); ++h) {
            if (!clusters[h] || clusters[h]->minimum == -1) continue;
            vector<int> child;
            clusters[h]->enumerate(child);
            for (int l : child) out.push_back(h * ru + l);
        }
    }
};

// ----------------------------- Arbor Porphyriana ------------------------------
struct Arbor {
    vector<vector<int>> adj;                    // adjacency list (undirected)
    unordered_map<string,int> id_of;            // label -> id
    vector<string> label_of;                    // id -> label

    unique_ptr<Van_Emde_Boas> veb;              // VEB index
    int U;                                      // capacity / universe size

    explicit Arbor(int universe_size = 128): U(universe_size) {
        veb = std::make_unique<Van_Emde_Boas>(U);
    }

    int ensure_node(const string& label) {
        auto it = id_of.find(label);
        if (it != id_of.end()) return it->second;
        int id = (int)label_of.size();
        if (id >= U) throw runtime_error("Out of VEB universe capacity. Increase U.");
        id_of[label] = id;
        label_of.push_back(label);
        if ((int)adj.size() <= id) adj.resize(id + 1);
        veb->insert(id);
        return id;
    }

    void connect_parent_child(const string& parent, const string& child) {
        int p = ensure_node(parent);
        int c = ensure_node(child);
        adj[p].push_back(c);
        adj[c].push_back(p);
    }

    // Dijkstra for unit weights (BFS equivalent but kept as requested).
    vector<int> shortest_path(const string& a, const string& b) const {
        auto ita = id_of.find(a), itb = id_of.find(b);
        if (ita == id_of.end() || itb == id_of.end()) return {};
        int s = ita->second, t = itb->second;
        int n = (int)label_of.size();
        const int INF = 1e9;
        vector<int> dist(n, INF), parent(n, -1);
        using P = pair<int,int>; // (dist, node)
        priority_queue<P, vector<P>, greater<P>> pq;
        dist[s] = 0; pq.push({0, s});
        while(!pq.empty()){
            auto [d,u] = pq.top(); pq.pop();
            if (d != dist[u]) continue;
            if (u == t) break;
            for(int v: adj[u]){
                if (dist[v] > d + 1){
                    dist[v] = d + 1;
                    parent[v] = u;
                    pq.push({dist[v], v});
                }
            }
        }
        if (dist[t] == INF) return {};
        vector<int> path;
        for (int cur = t; cur != -1; cur = parent[cur]) path.push_back(cur);
        reverse(path.begin(), path.end());
        return path;
    }

    void dump_veb_view() const {
        cout << "\n--- VEB View (U=" << U << ") ---\n";
        vector<int> keys; veb->enumerate(keys);
        sort(keys.begin(), keys.end()); keys.erase(unique(keys.begin(), keys.end()), keys.end());
        int ru = (int)ceil(sqrt((double)veb->universe_size));
        vector<vector<int>> by_cluster(ru);
        for (int k : keys) {
            int h = veb->high(k);
            if (h>=0 && h<ru) by_cluster[h].push_back(k);
        }
        for (int h = 0; h < ru; ++h) {
            if (by_cluster[h].empty()) continue;
            cout << "cluster[" << h << "] -> IDs: ";
            for (int k : by_cluster[h]) cout << k << ' ';
            cout << "\nlabels: ";
            for (int k : by_cluster[h]) {
                if (k >= 0 && k < (int)label_of.size()) cout << label_of[k] << ", ";
                else cout << "(unused:#" << k << "), ";
            }
            cout << "\n";
        }
        cout << "minID=" << veb->minimum << ", maxID=" << veb->maximum << "\n";
    }
};

// ----------------------------- Sample Builders -------------------------------
void build_sample_animals(Arbor& A){
    A.connect_parent_child("substance", "body");
    A.connect_parent_child("substance", "incorporeal");   // optional sibling

    // Body splits
    A.connect_parent_child("body", "living");
    A.connect_parent_child("body", "non_living");

    // Living splits
    A.connect_parent_child("living", "animal");
    A.connect_parent_child("living", "plant");

    // Animal splits by differentia
    A.connect_parent_child("animal", "rational_animal");
    A.connect_parent_child("animal", "irrational_animal");

    // Species under rational animal
    A.connect_parent_child("rational_animal", "man");
    A.connect_parent_child("rational_animal", "immortal_rational_animal");
    // (You could add "immortal_rational_animal" here if you want the classic ladder)

    // Individuals under man
    A.connect_parent_child("man", "Plato");
    A.connect_parent_child("man", "Socrates");
    A.connect_parent_child("man", "Aristotle");

    // A few species under irrational animal (for contrast)
    A.connect_parent_child("irrational_animal", "equine");
    A.connect_parent_child("irrational_animal", "canine");
    A.connect_parent_child("irrational_animal", "bird");
    // Kinds of bird
    //A.connect_parent_child("bird", "non-domestic");
    //A.connect_parent_child("bird", "domestic");
    // An example of bird
    A.connect_parent_child("bird", "chicken");
}

// Synthetic N-level Porphyrian-style tree with branching factor B.
void build_synthetic_porhyry(Arbor& A, int levels, int B){
    if (levels <= 0) return;
    vector<string> prev;
    string root = "L1_0";
    A.ensure_node(root);
    prev.push_back(root);
    for (int lvl = 2; lvl <= levels; ++lvl) {
        vector<string> cur;
        for (const string& p : prev) {
            for (int b = 0; b < B; ++b) {
                string name = string("L") + to_string(lvl) + "_" + to_string((int)cur.size());
                A.connect_parent_child(p, name);
                cur.push_back(name);
            }
        }
        prev.swap(cur);
    }
}

// ------------------------------ Diagram Utils --------------------------------
static string join_labels(const vector<int>& ids, const vector<string>& labels, const string& sep = " -> "){
    string out;
    for (size_t i=0;i<ids.size();++i){
        int id = ids[i];
        out += (id>=0 && id<(int)labels.size()) ? labels[id] : (string("#")+to_string(id));
        if (i+1<ids.size()) out += sep;
    }
    return out;
}

// ASCII tree printing from a chosen root label (ASCII-only connectors).
void print_ascii_tree_from_root(const Arbor& A, const string& root_lbl){
    auto it = A.id_of.find(root_lbl);
    if (it == A.id_of.end()) { cerr << "[diagram] root label not found: " << root_lbl << "\n"; return; }
    int root = it->second;

    function<void(int,int,string,bool)> dfs = [&](int u, int parent, string prefix, bool last){
        cout << prefix;
        if (!prefix.empty()) cout << (last ? "+-" : "+-");
        cout << A.label_of[u] << "\n";

        vector<int> children = A.adj[u];
        if (parent != -1) children.erase(remove(children.begin(), children.end(), parent), children.end());
        for (size_t i=0;i<children.size();++i){
            bool is_last = (i+1==children.size());
            string next_prefix = prefix + (prefix.empty()? "" : (last? "  " : "| "));
            dfs(children[i], u, next_prefix, is_last);
        }
    };

    dfs(root, -1, "", true);
}

// Graphviz DOT emitter (undirected). Writes to porphyry.dot
void emit_graphviz(const Arbor& A, const string& filename){
    ofstream ofs(filename);
    if (!ofs) { cerr << "[graphviz] cannot open: " << filename << "\n"; return; }
    ofs << "graph Porphyry {\n";
    ofs << "  rankdir=TB;\n";
    ofs << "  node [shape=box, style=rounded];\n";
    // Declare nodes
    for (size_t i=0;i<A.label_of.size();++i){
        ofs << "  n" << i << " [label=\"" << A.label_of[i] << "\"];\n";
    }
    // Undirected edges, avoid duplicates by u<v
    for (size_t u=0; u<A.adj.size(); ++u){
        for (int v : A.adj[u]) if ((int)u < v){
            ofs << "  n" << u << " -- n" << v << ";\n";
        }
    }
    ofs << "}\n";
    cerr << "[graphviz] wrote " << filename << " (render with: dot -Tpng " << filename << " -o porphyry.png)\n";
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Universe size: set comfortably larger than total nodes you plan to add.
    Arbor arbor(/*U=*/256);

    // --- Measure build time for the sample animal taxonomy ---
    auto t_build0 = high_resolution_clock::now();
    build_sample_animals(arbor);
    auto t_build1 = high_resolution_clock::now();
    auto build_us = duration_cast<microseconds>(t_build1 - t_build0).count();

    cout << "Build time (sample animals): " << build_us << " us\n";

    // --- VEB view ---
    arbor.dump_veb_view();

    // --- ASCII tree diagram (rooted at "entity") ---
    cout << "\nASCII Diagram (root=entity)\n";
    print_ascii_tree_from_root(arbor, "entity");

    // --- Graphviz DOT output ---
    emit_graphviz(arbor, "porphyry.dot");

    // --- Measure Dijkstra time for a sample query ---
    auto t_d0 = high_resolution_clock::now();
    auto path = arbor.shortest_path("Plato", "chicken");
    auto t_d1 = high_resolution_clock::now();
    auto dijk_us = duration_cast<microseconds>(t_d1 - t_d0).count();

    if (path.empty()) {
        cout << "\nNo path found between Plato and a featherless chicken\n";
    } else {
        cout << "\nShortest path (Plato -> chicken):\n  " << join_labels(path, arbor.label_of) << "\n";
        int edges = (int)path.size() - 1;
        int nodes_between = max(0, (int)path.size() - 2);
        cout << "Edges (hops): " << edges << "\n";
        cout << "Nodes between terms (excluding endpoints): " << nodes_between << "\n";
        cout << "Dijkstra time: " << dijk_us << " us\n";
    }

    return 0;
}