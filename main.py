from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict, Any
import json
import os


def _ask_int(prompt: str, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    while True:
        s = input(prompt).strip()
        try:
            x = int(s)
        except ValueError:
            print("Введите целое число.")
            continue
        if min_value is not None and x < min_value:
            print(f"Число должно быть >= {min_value}.")
            continue
        if max_value is not None and x > max_value:
            print(f"Число должно быть <= {max_value}.")
            continue
        return x


def _ask_yes_no(prompt: str) -> bool:
    while True:
        s = input(prompt).strip().lower()
        if s in ("y", "yes", "д", "да"):
            return True
        if s in ("n", "no", "н", "нет"):
            return False
        print("Введите да/нет (y/n).")


def _ask_choice(prompt: str, choices: List[str]) -> str:
    choices_l = [c.lower() for c in choices]
    while True:
        s = input(prompt).strip().lower()
        if s in choices_l:
            return s
        print(f"Введите один из вариантов: {', '.join(choices)}")


def _parse_ints_line(line: str) -> Optional[List[int]]:
    try:
        return [int(x) for x in line.strip().split()]
    except ValueError:
        return None


@dataclass
class Graph:
    directed: bool = False
    # В памяти храним как список смежности (множества)
    adj: List[Set[int]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.adj)

    def init_empty(self, n: int) -> None:
        self.adj = [set() for _ in range(n)]

    def add_edge(self, u: int, v: int) -> None:
        if not (0 <= u < self.n and 0 <= v < self.n):
            return
        self.adj[u].add(v)
        if not self.directed and u != v:
            self.adj[v].add(u)

    def remove_edge(self, u: int, v: int) -> None:
        if not (0 <= u < self.n and 0 <= v < self.n):
            return
        self.adj[u].discard(v)
        if not self.directed and u != v:
            self.adj[v].discard(u)

    def normalize(self) -> None:
        """Для неориентированного — сделать список смежности симметричным."""
        if self.directed:
            return
        for u in range(self.n):
            for v in list(self.adj[u]):
                self.adj[v].add(u)

    # --- Конвертации ---
    @staticmethod
    def from_adj_matrix(mat: List[List[int]], directed: bool) -> Graph:
        n = len(mat)
        if any(len(row) != n for row in mat):
            raise ValueError("Матрица смежности должна быть N x N.")
        g = Graph(directed=directed)
        g.init_empty(n)
        for i in range(n):
            for j in range(n):
                if mat[i][j] != 0:
                    g.add_edge(i, j)
        g.normalize()
        return g

    def to_adj_matrix(self) -> List[List[int]]:
        n = self.n
        mat = [[0] * n for _ in range(n)]
        for u in range(n):
            for v in self.adj[u]:
                mat[u][v] = 1
        if not self.directed:
            for i in range(n):
                for j in range(n):
                    if mat[i][j] or mat[j][i]:
                        mat[i][j] = mat[j][i] = 1
        return mat

    @staticmethod
    def from_adj_list(adj_list: List[List[int]], directed: bool) -> Graph:
        n = len(adj_list)
        g = Graph(directed=directed)
        g.init_empty(n)
        for u in range(n):
            for v in adj_list[u]:
                g.add_edge(u, v)
        g.normalize()
        return g

    def to_adj_list(self) -> List[List[int]]:
        return [sorted(nei) for nei in self.adj]

    @staticmethod
    def from_incidence(mat: List[List[int]], directed: bool | None = None) -> Graph:
        """
        Матрица инцидентности V x E.
        - НЕориентированный: 0/1 (или 2 для петли)
        - Ориентированный: -1 (откуда) и +1 (куда)
        Если directed=None: автоопределение по наличию -1.
        """
        if not mat:
            return Graph(directed=False, adj=[])
        V = len(mat)
        E = len(mat[0]) if V > 0 else 0
        for row in mat:
            if len(row) != E:
                raise ValueError("Матрица инцидентности должна быть прямоугольной.")

        auto_directed = any(any(x < 0 for x in row) for row in mat)
        if directed is None:
            directed = auto_directed

        g = Graph(directed=directed)
        g.init_empty(V)

        for e in range(E):
            col = [mat[v][e] for v in range(V)]
            if directed:
                tails = [v for v, x in enumerate(col) if x == -1]
                heads = [v for v, x in enumerate(col) if x == 1]
                if len(tails) == 1 and len(heads) == 1:
                    g.add_edge(tails[0], heads[0])
                else:
                    # запасной вариант: как неориентированное
                    ones = [v for v, x in enumerate(col) if x != 0]
                    if len(ones) == 2:
                        g.add_edge(ones[0], ones[1])
                    elif len(ones) == 1:
                        g.add_edge(ones[0], ones[0])
            else:
                ones = [v for v, x in enumerate(col) if x != 0]
                if len(ones) == 2:
                    g.add_edge(ones[0], ones[1])
                elif len(ones) == 1:
                    g.add_edge(ones[0], ones[0])

        g.normalize()
        return g

    def to_edge_list(self) -> List[Tuple[int, int]]:
        edges: List[Tuple[int, int]] = []
        if self.directed:
            for u in range(self.n):
                for v in sorted(self.adj[u]):
                    edges.append((u, v))
        else:
            seen = set()
            for u in range(self.n):
                for v in sorted(self.adj[u]):
                    a, b = (u, v) if u <= v else (v, u)
                    if (a, b) not in seen:
                        seen.add((a, b))
                        edges.append((a, b))
        return edges

    def to_incidence(self) -> List[List[int]]:
        V = self.n
        edges = self.to_edge_list()
        E = len(edges)
        mat = [[0] * E for _ in range(V)]
        for e, (u, v) in enumerate(edges):
            if self.directed:
                mat[u][e] = -1
                mat[v][e] = 1
            else:
                if u == v:
                    mat[u][e] = 2  # петля
                else:
                    mat[u][e] = 1
                    mat[v][e] = 1
        return mat

    # --- Файл (JSON) ---
    def to_dict(self) -> Dict[str, Any]:
        return {"directed": self.directed, "n": self.n, "adj": [sorted(list(s)) for s in self.adj]}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Graph:
        directed = bool(d.get("directed", False))
        adj_raw = d.get("adj", [])
        if not isinstance(adj_raw, list):
            raise ValueError("Неверный формат файла.")
        adj_list: List[List[int]] = []
        for row in adj_raw:
            adj_list.append([int(x) for x in row] if isinstance(row, list) else [])
        return Graph.from_adj_list(adj_list, directed=directed)


def print_adj_list(g: Graph) -> None:
    print("\nСписок смежности (вершины 1..N):")
    for i, nei in enumerate(g.to_adj_list(), start=1):
        print(f"{i}: " + (" ".join(str(v + 1) for v in nei) if nei else "-"))


def print_matrix(mat: List[List[int]], title: str) -> None:
    print(f"\n{title}:")
    if not mat:
        print("(пусто)")
        return
    cols = len(mat[0])
    width = max(2, max(len(str(x)) for row in mat for x in row))
    header = " " * (width + 1) + " ".join(f"{j+1:>{width}}" for j in range(cols))
    print(header)
    for i, row in enumerate(mat, start=1):
        print(f"{i:>{width}} " + " ".join(f"{x:>{width}}" for x in row))


def _matrix_correction(mat: List[List[int]], allowed_values: Optional[Set[int]]) -> List[List[int]]:
    if not mat:
        return mat
    while True:
        print_matrix(mat, "Введенная матрица")
        if not _ask_yes_no("Исправить значения? (y/n): "):
            return mat
        print("Введите: строка столбец новое_значение (нумерация с 1).")
        print("Или 'done' чтобы закончить исправление.")
        while True:
            s = input("> ").strip().lower()
            if s in ("done", "готово", "stop", "q"):
                break
            parts = s.split()
            if len(parts) != 3:
                print("Нужно 3 значения: i j val.")
                continue
            try:
                i = int(parts[0]) - 1
                j = int(parts[1]) - 1
                val = int(parts[2])
            except ValueError:
                print("i, j, val должны быть целыми.")
                continue
            if i < 0 or i >= len(mat) or j < 0 or (len(mat[0]) > 0 and j >= len(mat[0])):
                print("i/j вне диапазона.")
                continue
            if allowed_values is not None and val not in allowed_values:
                print(f"Разрешенные значения: {sorted(allowed_values)}")
                continue
            mat[i][j] = val


def input_adj_matrix() -> List[List[int]]:
    n = _ask_int("Введите количество вершин N: ", min_value=1, max_value=200)
    print("Введите матрицу смежности N x N (0/1), строки через пробел.")
    mat: List[List[int]] = []
    for i in range(n):
        while True:
            line = input(f"Строка {i+1}: ")
            nums = _parse_ints_line(line)
            if nums is None or len(nums) != n:
                print(f"Нужно {n} целых чисел.")
                continue
            if any(x not in (0, 1) for x in nums):
                print("Матрица смежности: только 0 или 1.")
                continue
            mat.append(nums)
            break
    return _matrix_correction(mat, allowed_values={0, 1})


def input_incidence_matrix() -> List[List[int]]:
    v = _ask_int("Введите количество вершин V (строки): ", min_value=1, max_value=200)
    e = _ask_int("Введите количество ребер E (столбцы): ", min_value=0, max_value=500)
    print("Введите матрицу инцидентности V x E.")
    print("НЕориентированный: обычно 0/1 (в каждом столбце две 1).")
    print("Ориентированный: -1 (откуда) и +1 (куда) в каждом столбце.")
    allowed = {-1, 0, 1, 2}
    mat: List[List[int]] = []
    for i in range(v):
        while True:
            if e == 0:
                mat.append([])
                break
            line = input(f"Строка {i+1}: ")
            nums = _parse_ints_line(line)
            if nums is None or len(nums) != e:
                print(f"Нужно {e} целых чисел.")
                continue
            if any(x not in allowed for x in nums):
                print("Допустимые значения: -1, 0, 1 (и 2 для петли).")
                continue
            mat.append(nums)
            break
    return _matrix_correction(mat, allowed_values=allowed)


def input_adj_list() -> List[List[int]]:
    n = _ask_int("Введите количество вершин N: ", min_value=1, max_value=200)
    print("Введите список смежности.")
    print("Для каждой вершины: перечислите соседей через пробел (номера 1..N). Пусто или 0 — нет соседей.")
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        while True:
            line = input(f"Соседи вершины {i+1}: ").strip()
            if line == "" or line == "0":
                adj[i] = []
                break
            nums = _parse_ints_line(line)
            if nums is None:
                print("Введите числа через пробел.")
                continue
            cleaned = []
            ok = True
            for x in nums:
                if x == 0:
                    continue
                if x < 1 or x > n:
                    ok = False
                    break
                cleaned.append(x - 1)
            if not ok:
                print(f"Соседи должны быть в диапазоне 1..{n}.")
                continue
            adj[i] = cleaned
            break

    while True:
        print("\nВведенный список смежности:")
        for i, row in enumerate(adj, start=1):
            print(f"{i}: " + (" ".join(str(x + 1) for x in row) if row else "-"))
        if not _ask_yes_no("Исправить список смежности? (y/n): "):
            return adj
        v = _ask_int("Какую вершину исправить? (1..N): ", min_value=1, max_value=n) - 1
        line = input(f"Новые соседи для {v+1} (через пробел, 0/пусто — очистить): ").strip()
        if line == "" or line == "0":
            adj[v] = []
            continue
        nums = _parse_ints_line(line)
        if nums is None:
            print("Некорректный ввод.")
            continue
        cleaned = []
        ok = True
        for x in nums:
            if x == 0:
                continue
            if x < 1 or x > n:
                ok = False
                break
            cleaned.append(x - 1)
        if not ok:
            print(f"Соседи должны быть в диапазоне 1..{n}.")
            continue
        adj[v] = cleaned


def main() -> None:
    g = Graph(directed=False)
    print("Лабораторная: Хранение графа в памяти (Python)")
    print("Внутри программа хранит граф как список смежности.\n")

    while True:
        print("\n=== МЕНЮ ===")
        print("1) Ввести граф")
        print("2) Показать граф")
        print("3) Добавить/удалить ребро")
        print("4) Сохранить в файл (JSON)")
        print("5) Загрузить из файла (JSON)")
        print(f"6) Сменить тип графа (сейчас: {'ориентированный' if g.directed else 'неориентированный'})")
        print("0) Выход")

        cmd = _ask_choice("Выберите пункт: ", ["0", "1", "2", "3", "4", "5", "6"])
        if cmd == "0":
            break

        if cmd == "6":
            g.directed = not g.directed
            g.normalize()
            print("Готово.")
            continue

        if cmd == "1":
            print("\nФорма ввода:")
            print("a) Список смежности")
            print("b) Матрица смежности")
            print("c) Матрица инцидентности")
            form = _ask_choice("Выберите (a/b/c): ", ["a", "b", "c"])

            try:
                if form == "a":
                    adj = input_adj_list()
                    g = Graph.from_adj_list(adj, directed=g.directed)
                elif form == "b":
                    mat = input_adj_matrix()
                    g = Graph.from_adj_matrix(mat, directed=g.directed)
                else:
                    mat = input_incidence_matrix()
                    auto_dir = any(any(x < 0 for x in row) for row in mat)
                    g = Graph.from_incidence(mat, directed=True if auto_dir else g.directed)

                print("\nГраф введён и сохранён в памяти (как список смежности).")
                print_adj_list(g)
            except Exception as e:
                print(f"Ошибка: {e}")
            continue

        if cmd == "2":
            if g.n == 0:
                print("Граф пустой. Сначала введите граф.")
                continue
            print("\nФорма вывода:")
            print("a) Список смежности")
            print("b) Матрица смежности")
            print("c) Матрица инцидентности")
            form = _ask_choice("Выберите (a/b/c): ", ["a", "b", "c"])
            if form == "a":
                print_adj_list(g)
            elif form == "b":
                print_matrix(g.to_adj_matrix(), "Матрица смежности")
            else:
                print_matrix(g.to_incidence(), "Матрица инцидентности (V x E)")
            continue

        if cmd == "3":
            if g.n == 0:
                print("Граф пустой. Сначала введите граф.")
                continue
            print("\nОперация:")
            print("a) Добавить ребро")
            print("b) Удалить ребро")
            op = _ask_choice("Выберите (a/b): ", ["a", "b"])
            u = _ask_int(f"u (1..{g.n}): ", min_value=1, max_value=g.n) - 1
            v = _ask_int(f"v (1..{g.n}): ", min_value=1, max_value=g.n) - 1
            if op == "a":
                g.add_edge(u, v)
            else:
                g.remove_edge(u, v)
            g.normalize()
            print_adj_list(g)
            continue

        if cmd == "4":
            if g.n == 0:
                print("Граф пустой. Нечего сохранять.")
                continue
            path = input("Имя файла (например graph.json): ").strip()
            if not path:
                print("Пустое имя файла.")
                continue
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(g.to_dict(), f, ensure_ascii=False, indent=2)
                print(f"Сохранено в {os.path.abspath(path)}")
            except Exception as e:
                print(f"Ошибка сохранения: {e}")
            continue

        if cmd == "5":
            path = input("Имя файла (например graph.json): ").strip()
            if not path:
                print("Пустое имя файла.")
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                g = Graph.from_dict(d)
                print("Загружено.")
                print_adj_list(g)
            except Exception as e:
                print(f"Ошибка загрузки: {e}")
            continue


if __name__ == "__main__":
    main()