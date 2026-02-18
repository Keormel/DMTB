from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Literal

GraphType = Literal["directed", "undirected"]
IncidenceMode = Literal["directed_pm1", "undirected_1_2"]


def ask_int(prompt: str, lo: int | None = None, hi: int | None = None) -> int:
    while True:
        s = input(prompt).strip()
        try:
            x = int(s)
        except ValueError:
            print("Введите целое число.")
            continue
        if lo is not None and x < lo:
            print(f"Число должно быть >= {lo}")
            continue
        if hi is not None and x > hi:
            print(f"Число должно быть <= {hi}")
            continue
        return x


def ask_choice(prompt: str, choices: List[str]) -> str:
    choices_set = {c.lower() for c in choices}
    while True:
        s = input(prompt).strip().lower()
        if s in choices_set:
            return s
        print(f"Введите один из вариантов: {', '.join(choices)}")


def parse_ints(line: str) -> List[int]:
    line = line.replace(",", " ")
    parts = [p for p in line.split() if p]
    out = []
    for p in parts:
        out.append(int(p))
    return out


@dataclass
class Graph:
    n: int
    graph_type: GraphType = "directed"
    adj: List[Set[int]] = field(default_factory=list)  # 0-based

    def __post_init__(self) -> None:
        if not self.adj:
            self.adj = [set() for _ in range(self.n)]
        if len(self.adj) != self.n:
            raise ValueError("Размер списка смежности не совпадает с n")

    def _check(self, v: int) -> None:
        if not (0 <= v < self.n):
            raise ValueError(f"Вершина {v+1} вне диапазона 1..{self.n}")

    def add_arc(self, u: int, v: int) -> None:
        self._check(u)
        self._check(v)
        self.adj[u].add(v)
        if self.graph_type == "undirected" and u != v:
            self.adj[v].add(u)

    def remove_arc(self, u: int, v: int) -> None:
        self._check(u)
        self._check(v)
        self.adj[u].discard(v)
        if self.graph_type == "undirected" and u != v:
            self.adj[v].discard(u)

    def to_edge_list(self) -> List[Tuple[int, int]]:
        edges: List[Tuple[int, int]] = []
        if self.graph_type == "directed":
            for u in range(self.n):
                for v in sorted(self.adj[u]):
                    edges.append((u, v))
        else:
            seen = set()
            for u in range(self.n):
                for v in self.adj[u]:
                    a, b = (u, v) if u <= v else (v, u)
                    if (a, b) not in seen:
                        seen.add((a, b))
                        edges.append((a, b))
            edges.sort()
        return edges

    def to_adj_matrix(self) -> List[List[int]]:
        M = [[0] * self.n for _ in range(self.n)]
        for u in range(self.n):
            for v in self.adj[u]:
                M[u][v] = 1
        if self.graph_type == "undirected":
            for i in range(self.n):
                for j in range(self.n):
                    if M[i][j] or M[j][i]:
                        M[i][j] = M[j][i] = 1
        return M

    @classmethod
    def from_adj_matrix(cls, M: List[List[int]], graph_type: GraphType) -> "Graph":
        n = len(M)
        if any(len(row) != n for row in M):
            raise ValueError("Матрица должна быть квадратной NxN")
        g = cls(n=n, graph_type=graph_type)
        for i in range(n):
            for j in range(n):
                if M[i][j] != 0:
                    g.add_arc(i, j)
        return g

    def to_incidence_matrix(self, mode: IncidenceMode) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
        edges = self.to_edge_list()
        E = len(edges)
        Inc = [[0] * E for _ in range(self.n)]

        for e, (u, v) in enumerate(edges):
            if mode == "directed_pm1":
                if u == v:
                    # петля в направленном графе: ставим 2
                    Inc[u][e] = 2
                else:
                    Inc[u][e] = -1
                    Inc[v][e] = 1
            else:  # undirected_1_2
                if u == v:
                    Inc[u][e] = 2
                else:
                    Inc[u][e] = 1
                    Inc[v][e] = 1
        return Inc, edges

    @classmethod
    def from_incidence_matrix(cls, Inc: List[List[int]], graph_type: GraphType, mode: IncidenceMode) -> "Graph":
        V = len(Inc)
        if V == 0:
            raise ValueError("Пустая матрица")
        E = len(Inc[0])
        if any(len(row) != E for row in Inc):
            raise ValueError("Все строки должны быть одинаковой длины")

        edges: List[Tuple[int, int]] = []

        if mode == "directed_pm1":
            # в каждом столбце: одна -1 и одна +1
            for e in range(E):
                u = v = None
                for i in range(V):
                    val = Inc[i][e]
                    if val == -1:
                        u = i
                    elif val == 1:
                        v = i
                    elif val != 0:
                        raise ValueError("В directed_pm1 допускаются только -1, 0, 1")
                if u is None or v is None:
                    raise ValueError(f"Столбец {e+1} не содержит пары (-1,+1)")
                edges.append((u, v))

        else:  # undirected_1_2
            for e in range(E):
                ones = []
                loop = None
                for i in range(V):
                    val = Inc[i][e]
                    if val == 1:
                        ones.append(i)
                    elif val == 2:
                        loop = i
                    elif val != 0:
                        raise ValueError("В undirected_1_2 допускаются только 0, 1, 2")
                if loop is not None:
                    if ones:
                        raise ValueError(f"Столбец {e+1}: нельзя иметь и 2 и 1 одновременно")
                    edges.append((loop, loop))
                else:
                    if len(ones) != 2:
                        raise ValueError(f"Столбец {e+1}: должно быть ровно две '1' (или одна '2' для петли)")
                    edges.append((ones[0], ones[1]))

        g = cls(n=V, graph_type=graph_type)
        for u, v in edges:
            g.add_arc(u, v)
        return g

    def pretty_adj_list(self) -> str:
        lines = []
        for u in range(self.n):
            neigh = " ".join(str(v + 1) for v in sorted(self.adj[u]))
            lines.append(f"{u+1}: {neigh}".rstrip())
        return "\n".join(lines)

    @staticmethod
    def pretty_matrix(M: List[List[int]]) -> str:
        return "\n".join(" ".join(f"{x:3d}" for x in row) for row in M)

    @staticmethod
    def pretty_incidence_matrix(Inc: List[List[int]], edges: List[Tuple[int, int]]) -> str:
        """Вывод матрицы инцидентности с заголовками E и V"""
        V = len(Inc)
        E = len(edges) if Inc and len(Inc) > 0 else 0
        
        if V == 0 or E == 0:
            return "Пустая матрица"
        
        # Заголовок для рёбер
        header = "     " + " ".join(f"E{i+1:2d}" for i in range(E))
        lines = [header]
        
        # Строки с вершинами
        for i in range(V):
            row_str = f"V{i+1:2d}  " + " ".join(f"{Inc[i][j]:3d}" for j in range(E))
            lines.append(row_str)
        
        return "\n".join(lines)

    @staticmethod
    def pretty_edges(edges: List[Tuple[int, int]]) -> str:
        out = []
        for i, (u, v) in enumerate(edges, start=1):
            out.append(f"E{i}: ({u+1}, {v+1})")
        return "\n".join(out)


# --------------------------
# ВВОДЫ ОТ ПОЛЬЗОВАТЕЛЯ
# --------------------------
def input_adj_list() -> Graph:
    n = ask_int("Сколько вершин N? ", 1, 200)
    gt = ask_choice("Граф directed или undirected? (directed/undirected): ", ["directed", "undirected"])
    g = Graph(n=n, graph_type=gt)  # пустой

    print("\nВвод списка смежности (вершины 1..N).")
    print("Формат строки: соседи через пробел. Пустая строка = нет соседей.")
    print("Пример: для вершины 1: 2 4 6\n")

    for u in range(n):
        line = input(f"{u+1}: ").strip()
        if not line:
            continue
        nums = parse_ints(line)
        for x in nums:
            if not (1 <= x <= n):
                raise ValueError(f"Ошибка: в строке {u+1} вершина {x} вне 1..{n}")
            g.add_arc(u, x - 1)

    return g


def input_adj_matrix() -> Graph:
    n = ask_int("Сколько вершин N? ", 1, 200)
    gt = ask_choice("Граф directed или undirected? (directed/undirected): ", ["directed", "undirected"])
    print("\nВвод матрицы смежности NxN (0/1). Вводи строки через пробел.\n")

    M: List[List[int]] = []
    for i in range(n):
        while True:
            line = input(f"Строка {i+1}: ").strip()
            try:
                row = parse_ints(line)
            except Exception:
                print("Ошибка чтения. Введите числа 0/1 через пробел.")
                continue
            if len(row) != n:
                print(f"Нужно {n} чисел.")
                continue
            if any(x not in (0, 1) for x in row):
                print("Только 0 или 1.")
                continue
            M.append(row)
            break

    return Graph.from_adj_matrix(M, graph_type=gt)


def input_incidence_matrix() -> Graph:
    n = ask_int("Сколько вершин V? ", 1, 200)
    e = ask_int("Сколько рёбер/дуг E? ", 0, 5000)

    gt = ask_choice("Граф directed или undirected? (directed/undirected): ", ["directed", "undirected"])
    mode: IncidenceMode = "directed_pm1" if gt == "directed" else "undirected_1_2"

    print("\nВвод матрицы инцидентности VxE.")
    if mode == "directed_pm1":
        print("Формат directed_pm1: в каждом столбце одна -1 (откуда) и одна +1 (куда). Остальные 0.")
        print("Петли в этом режиме не поддерживаем (лучше вводить через матрицу смежности).")
    else:
        print("Формат undirected_1_2: в столбце две '1' (концы ребра) или одна '2' (петля).")

    Inc: List[List[int]] = []
    for i in range(n):
        while True:
            line = input(f"Строка {i+1}: ").strip()
            row = parse_ints(line) if line else []
            if len(row) != e:
                print(f"Нужно ровно {e} чисел.")
                continue
            allowed = (-1, 0, 1) if mode == "directed_pm1" else (0, 1, 2)
            if any(x not in allowed for x in row):
                print(f"Недопустимые значения. Разрешено: {allowed}")
                continue
            Inc.append(row)
            break

    return Graph.from_incidence_matrix(Inc, graph_type=gt, mode=mode)


def input_graph_menu() -> Graph:
    print("\nКак вы хотите ввести граф?")
    print("1) Список смежности")
    print("2) Матрица смежности")
    print("3) Матрица инцидентности")
    t = ask_int("Выбор: ", 1, 3)

    if t == 1:
        return input_adj_list()
    if t == 2:
        return input_adj_matrix()
    return input_incidence_matrix()


# --------------------------
# МЕНЮ
# --------------------------
def main() -> None:
    g: Graph | None = None

    while True:
        print("\n==================== МЕНЮ ====================")
        print("1) Ввести новый граф")
        print("2) Показать список смежности")
        print("3) Показать матрицу смежности")
        print("4) Показать матрицу инцидентности")
        print("5) Показать список рёбер/дуг")
        print("6) Добавить дугу/ребро")
        print("7) Удалить дугу/ребро")
        print("0) Выход")
        print("==============================================")
        cmd = ask_int("Команда: ", 0, 7)

        if cmd == 0:
            break

        if cmd == 1:
            try:
                g = input_graph_menu()
                print("\n✅ Граф загружен.")
            except Exception as ex:
                print(f"❌ Ошибка ввода: {ex}")
                g = None

        elif g is None:
            print("❗ Сначала введите граф (пункт 1).")

        elif cmd == 2:
            print("\nСписок смежности (1..N):")
            print(g.pretty_adj_list())

        elif cmd == 3:
            print("\nМатрица смежности:")
            A = g.to_adj_matrix()
            print(Graph.pretty_matrix(A))

        elif cmd == 4:
            mode: IncidenceMode = "directed_pm1" if g.graph_type == "directed" else "undirected_1_2"
            Inc, edges = g.to_incidence_matrix(mode=mode)
            print("\nРёбра/дуги:")
            print(Graph.pretty_edges(edges))
            print("\nМатрица инцидентности:")
            print(Graph.pretty_incidence_matrix(Inc, edges))

        elif cmd == 5:
            edges = g.to_edge_list()
            print("\nСписок рёбер/дуг:")
            print(Graph.pretty_edges(edges))

        elif cmd == 6:
            u = ask_int("Откуда (вершина u, 1..N): ", 1, g.n) - 1
            v = ask_int("Куда (вершина v, 1..N): ", 1, g.n) - 1
            g.add_arc(u, v)
            if g.graph_type == "undirected":
                print("✅ Ребро добавлено (u-v).")
            else:
                print("✅ Дуга добавлена (u->v).")

        elif cmd == 7:
            u = ask_int("Откуда (вершина u, 1..N): ", 1, g.n) - 1
            v = ask_int("Куда (вершина v, 1..N): ", 1, g.n) - 1
            g.remove_arc(u, v)
            if g.graph_type == "undirected":
                print("✅ Ребро удалено (u-v).")
            else:
                print("✅ Дуга удалена (u->v).")


if __name__ == "__main__":
    main()
