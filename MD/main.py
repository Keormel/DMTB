from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict, Any
import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import random
import time


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


class GraphGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.graph = Graph(directed=False)
        self.setup_ui()

    def setup_ui(self) -> None:
        """Создание графического интерфейса"""
        # Верхняя панель с кнопками
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Button(top_frame, text="📥 Ввести граф", command=self.input_graph).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="🎲 Сгенерировать", command=self.generate_graph_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="📊 Показать", command=self.show_graph).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="➕ Добавить ребро", command=self.add_edge_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="➖ Удалить ребро", command=self.remove_edge_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="💾 Сохранить", command=self.save_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="📂 Загрузить", command=self.load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_frame, text="🔄 Преобразование", command=self.show_transformation).pack(side=tk.LEFT, padx=2)

        # Кнопка переключения типа графа
        self.type_label = ttk.Label(top_frame, text="Неориентированный")
        self.type_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(top_frame, text="🔄 Сменить тип", command=self.toggle_graph_type).pack(side=tk.LEFT, padx=2)

        # Основная область с текстом
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Текстовое поле с прокруткой
        scrollbar = ttk.Scrollbar(content_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_output = tk.Text(content_frame, yscrollcommand=scrollbar.set, font=("Courier", 10))
        self.text_output.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.text_output.yview)

        self.text_output.insert(tk.END, "Лабораторная: Хранение графа в памяти (Python)\n")
        self.text_output.insert(tk.END, "Внутри программа хранит граф как список смежности.\n\n")
        self.text_output.config(state=tk.DISABLED)

    def log(self, text: str) -> None:
        """Добавить текст в вывод"""
        self.text_output.config(state=tk.NORMAL)
        self.text_output.insert(tk.END, text + "\n")
        self.text_output.see(tk.END)
        self.text_output.config(state=tk.DISABLED)

    def clear_output(self) -> None:
        """Очистить вывод"""
        self.text_output.config(state=tk.NORMAL)
        self.text_output.delete(1.0, tk.END)
        self.text_output.config(state=tk.DISABLED)

    def toggle_graph_type(self) -> None:
        """Переключить тип графа"""
        self.graph.directed = not self.graph.directed
        self.graph.normalize()
        self.type_label.config(text="Ориентированный" if self.graph.directed else "Неориентированный")
        self.log(f"Тип графа изменен на: {'Ориентированный' if self.graph.directed else 'Неориентированный'}")

    def input_graph(self) -> None:
        """Диалог ввода графа"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Ввести граф")
        dialog.geometry("400x250")

        ttk.Label(dialog, text="Выберите форму ввода:").pack(pady=10)

        ttk.Button(dialog, text="Список смежности", command=lambda: self.input_adj_list_dialog(dialog)).pack(pady=5)
        ttk.Button(dialog, text="Матрица смежности", command=lambda: self.input_adj_matrix_dialog(dialog)).pack(pady=5)
        ttk.Button(dialog, text="Матрица инцидентности", command=lambda: self.input_incidence_dialog(dialog)).pack(pady=5)

    def input_adj_list_dialog(self, parent: tk.Widget) -> None:
        """Диалог для ввода списка смежности"""
        parent.destroy()
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Ввести список смежности")
        dialog.geometry("600x500")

        # Верхняя часть
        top_frame = ttk.Frame(dialog)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(top_frame, text="Количество вершин N:").pack(side=tk.LEFT, padx=5)
        n_var = tk.StringVar(value="5")
        entry_n = ttk.Entry(top_frame, textvariable=n_var, width=10)
        entry_n.pack(side=tk.LEFT, padx=5)

        ttk.Label(dialog, text="Для каждой вершины введите соседей через пробел (номера 1..N, или оставьте пустым):").pack(pady=5)

        # Главный frame со скроллом
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Список для всех строк
        rows_frame = ttk.Frame(main_frame)
        rows_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Текстовое поле вместо множества Entry-й
        text_widget = tk.Text(rows_frame, height=15, width=50, yscrollcommand=scrollbar.set)
        text_widget.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        def process_input() -> None:
            try:
                n = int(n_var.get())
                if n < 1:
                    messagebox.showerror("Ошибка", "N должно быть >= 1")
                    return
                
                text_content = text_widget.get("1.0", tk.END).strip()
                lines = text_content.split('\n')
                
                if len(lines) < n:
                    messagebox.showerror("Ошибка", f"Нужно {n} строк, а введено {len(lines)}")
                    return
                
                adj_list: List[List[int]] = [[] for _ in range(n)]

                for i in range(n):
                    line = lines[i].strip()
                    if line == "":
                        adj_list[i] = []
                    else:
                        nums = [int(x) - 1 for x in line.split()]
                        if any(x < 0 or x >= n for x in nums):
                            messagebox.showerror("Ошибка", f"Строка {i+1}: соседи должны быть в диапазоне 1..{n}")
                            return
                        adj_list[i] = nums

                self.graph = Graph.from_adj_list(adj_list, directed=self.graph.directed)
                dialog.destroy()
                self.clear_output()
                self.log("✓ Граф введён и сохранён в памяти (как список смежности).")
                self.show_adj_list()
            except ValueError as e:
                messagebox.showerror("Ошибка", f"Некорректный ввод: {e}")

        bottom_frame = ttk.Frame(dialog)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        ttk.Button(bottom_frame, text="Готово", command=process_input).pack()

    def input_adj_matrix_dialog(self, parent: tk.Widget) -> None:
        """Диалог для ввода матрицы смежности"""
        parent.destroy()
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Ввести матрицу смежности")
        dialog.geometry("600x500")

        # Верхняя часть
        top_frame = ttk.Frame(dialog)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(top_frame, text="Количество вершин N:").pack(side=tk.LEFT, padx=5)
        n_var = tk.StringVar(value="5")
        entry_n = ttk.Entry(top_frame, textvariable=n_var, width=10)
        entry_n.pack(side=tk.LEFT, padx=5)

        ttk.Label(dialog, text="Введите матрицу смежности N x N (0 или 1, строки через Enter):").pack(pady=5)

        # Главный frame со скроллом
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Текстовое поле вместо множества Entry-й
        text_widget = tk.Text(main_frame, height=15, width=50, yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        def process_input() -> None:
            try:
                n = int(n_var.get())
                if n < 1:
                    messagebox.showerror("Ошибка", "N должно быть >= 1")
                    return
                
                text_content = text_widget.get("1.0", tk.END).strip()
                lines = text_content.split('\n')
                
                if len(lines) < n:
                    messagebox.showerror("Ошибка", f"Нужно {n} строк, а введено {len(lines)}")
                    return
                
                mat: List[List[int]] = []

                for i in range(n):
                    line = lines[i].strip()
                    if not line:
                        messagebox.showerror("Ошибка", f"Строка {i+1} пуста")
                        return
                    nums = [int(x) for x in line.split()]
                    if len(nums) != n:
                        messagebox.showerror("Ошибка", f"Строка {i+1} должна содержать {n} значений")
                        return
                    if any(x not in (0, 1) for x in nums):
                        messagebox.showerror("Ошибка", "Матрица должна содержать только 0 и 1")
                        return
                    mat.append(nums)

                self.graph = Graph.from_adj_matrix(mat, directed=self.graph.directed)
                dialog.destroy()
                self.clear_output()
                self.log("✓ Граф введён и сохранён в памяти (как список смежности).")
                self.show_adj_list()
            except ValueError as e:
                messagebox.showerror("Ошибка", f"Некорректный ввод: {e}")

        bottom_frame = ttk.Frame(dialog)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        ttk.Button(bottom_frame, text="Готово", command=process_input).pack()

    def input_incidence_dialog(self, parent: tk.Widget) -> None:
        """Диалог для ввода матрицы инцидентности"""
        parent.destroy()
        messagebox.showinfo("Информация", "Функция пока в разработке")

    def show_graph(self) -> None:
        """Показать граф"""
        if self.graph.n == 0:
            messagebox.showwarning("Ошибка", "Граф пустой. Сначала введите граф.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Показать граф")
        dialog.geometry("400x200")

        ttk.Label(dialog, text="Выберите форму вывода:").pack(pady=10)

        ttk.Button(dialog, text="Список смежности", command=lambda: [self.show_adj_list(), dialog.destroy()]).pack(pady=5)
        ttk.Button(dialog, text="Матрица смежности", command=lambda: [self.show_adj_matrix(), dialog.destroy()]).pack(pady=5)
        ttk.Button(dialog, text="Матрица инцидентности", command=lambda: [self.show_incidence_matrix(), dialog.destroy()]).pack(pady=5)

    def show_adj_list(self) -> None:
        """Показать список смежности"""
        self.clear_output()
        self.log("\n📋 Список смежности (вершины 1..N):")
        for i, nei in enumerate(self.graph.to_adj_list(), start=1):
            if nei:
                self.log(f"{i}: " + " ".join(str(v + 1) for v in nei))
            else:
                self.log(f"{i}: -")

    def show_adj_matrix(self) -> None:
        """Показать матрицу смежности"""
        self.clear_output()
        self.log("\n📊 Матрица смежности:")
        mat = self.graph.to_adj_matrix()
        if not mat:
            self.log("(пусто)")
            return

        cols = len(mat[0])
        width = 3
        header = "    " + " ".join(f"{j+1:>{width}}" for j in range(cols))
        self.log(header)

        for i, row in enumerate(mat, start=1):
            self.log(f"{i:>{3}} " + " ".join(f"{x:>{width}}" for x in row))

    def show_incidence_matrix(self) -> None:
        """Показать матрицу инцидентности"""
        self.clear_output()
        self.log("\n📊 Матрица инцидентности (V x E):")
        mat = self.graph.to_incidence()
        if not mat:
            self.log("(пусто)")
            return

        cols = len(mat[0]) if mat else 0
        width = 3
        header = "    " + " ".join(f"{j+1:>{width}}" for j in range(cols))
        self.log(header)

        for i, row in enumerate(mat, start=1):
            self.log(f"{i:>{3}} " + " ".join(f"{x:>{width}}" for x in row))

    def add_edge_dialog(self) -> None:
        """Диалог добавления ребра"""
        if self.graph.n == 0:
            messagebox.showwarning("Ошибка", "Граф пустой.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Добавить ребро")
        dialog.geometry("300x150")

        ttk.Label(dialog, text=f"u (1..{self.graph.n}):").pack(pady=5)
        u_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=u_var).pack(pady=5)

        ttk.Label(dialog, text=f"v (1..{self.graph.n}):").pack(pady=5)
        v_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=v_var).pack(pady=5)

        def add() -> None:
            try:
                u = int(u_var.get()) - 1
                v = int(v_var.get()) - 1
                if not (0 <= u < self.graph.n and 0 <= v < self.graph.n):
                    messagebox.showerror("Ошибка", f"Вершины должны быть в диапазоне 1..{self.graph.n}")
                    return
                self.graph.add_edge(u, v)
                self.graph.normalize()
                dialog.destroy()
                self.show_adj_list()
                messagebox.showinfo("Успех", f"Ребро ({u+1}, {v+1}) добавлено")
            except ValueError:
                messagebox.showerror("Ошибка", "Введите целые числа")

        ttk.Button(dialog, text="Добавить", command=add).pack(pady=10)

    def remove_edge_dialog(self) -> None:
        """Диалог удаления ребра"""
        if self.graph.n == 0:
            messagebox.showwarning("Ошибка", "Граф пустой.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Удалить ребро")
        dialog.geometry("300x150")

        ttk.Label(dialog, text=f"u (1..{self.graph.n}):").pack(pady=5)
        u_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=u_var).pack(pady=5)

        ttk.Label(dialog, text=f"v (1..{self.graph.n}):").pack(pady=5)
        v_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=v_var).pack(pady=5)

        def remove() -> None:
            try:
                u = int(u_var.get()) - 1
                v = int(v_var.get()) - 1
                if not (0 <= u < self.graph.n and 0 <= v < self.graph.n):
                    messagebox.showerror("Ошибка", f"Вершины должны быть в диапазоне 1..{self.graph.n}")
                    return
                self.graph.remove_edge(u, v)
                self.graph.normalize()
                dialog.destroy()
                self.show_adj_list()
                messagebox.showinfo("Успех", f"Ребро ({u+1}, {v+1}) удалено")
            except ValueError:
                messagebox.showerror("Ошибка", "Введите целые числа")

        ttk.Button(dialog, text="Удалить", command=remove).pack(pady=10)

    def save_file(self) -> None:
        """Сохранить в JSON файл"""
        if self.graph.n == 0:
            messagebox.showwarning("Ошибка", "Граф пустой. Нечего сохранять.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(self.graph.to_dict(), f, ensure_ascii=False, indent=2)
                messagebox.showinfo("Успех", f"Сохранено в {filepath}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка сохранения: {e}")

    def load_file(self) -> None:
        """Загрузить из JSON файла"""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.graph = Graph.from_dict(d)
                self.type_label.config(text="Ориентированный" if self.graph.directed else "Неориентированный")
                self.clear_output()
                self.log(f"✓ Загружено из {filepath}")
                self.show_adj_list()
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка загрузки: {e}")

    def generate_graph_dialog(self) -> None:
        """Диалог для генерации случайного графа"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Генерировать граф")
        dialog.geometry("350x250")

        ttk.Label(dialog, text="Количество вершин N:").pack(pady=5)
        n_var = tk.StringVar(value="5")
        ttk.Entry(dialog, textvariable=n_var, width=10).pack(pady=5)

        ttk.Label(dialog, text="Вероятность ребра (0.0 - 1.0):").pack(pady=5)
        prob_var = tk.StringVar(value="0.3")
        ttk.Entry(dialog, textvariable=prob_var, width=10).pack(pady=5)

        ttk.Label(dialog, text="Тип генерации:").pack(pady=5)
        gen_type = tk.StringVar(value="random")
        ttk.Radiobutton(dialog, text="Случайный граф", variable=gen_type, value="random").pack()
        ttk.Radiobutton(dialog, text="Полный граф", variable=gen_type, value="complete").pack()
        ttk.Radiobutton(dialog, text="Дерево", variable=gen_type, value="tree").pack()

        def generate() -> None:
            try:
                n = int(n_var.get())
                if n < 1 or n > 100:
                    messagebox.showerror("Ошибка", "N должно быть от 1 до 100")
                    return

                prob = float(prob_var.get())
                if not (0 <= prob <= 1):
                    messagebox.showerror("Ошибка", "Вероятность должна быть от 0 до 1")
                    return

                gtype = gen_type.get()

                # Генерируем граф
                adj_list: List[List[int]] = [[] for _ in range(n)]

                if gtype == "random":
                    for i in range(n):
                        for j in range(i + 1, n):
                            if random.random() < prob:
                                adj_list[i].append(j)
                                adj_list[j].append(i)
                elif gtype == "complete":
                    for i in range(n):
                        for j in range(i + 1, n):
                            adj_list[i].append(j)
                            adj_list[j].append(i)
                elif gtype == "tree":
                    # Простое дерево: каждая вершина i связана с вершиной i+1
                    for i in range(n - 1):
                        adj_list[i].append(i + 1)
                        adj_list[i + 1].append(i)

                # Сортируем список смежности
                for i in range(n):
                    adj_list[i].sort()

                self.graph = Graph.from_adj_list(adj_list, directed=self.graph.directed)
                dialog.destroy()
                self.clear_output()
                self.log(f"✓ Сгенерирован {gtype} граф с {n} вершинами")
                self.show_adj_list()
            except ValueError:
                messagebox.showerror("Ошибка", "Некорректный ввод")

        ttk.Button(dialog, text="Генерировать", command=generate).pack(pady=10)

    def show_transformation(self) -> None:
        """Показать процедуру преобразования графа между представлениями"""
        if self.graph.n == 0:
            messagebox.showwarning("Ошибка", "Граф пустой. Сначала введите или сгенерируйте граф.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Преобразование графа")
        dialog.geometry("800x600")

        # Текстовое поле с прокруткой
        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        output_text = tk.Text(text_frame, font=("Courier", 9), yscrollcommand=scrollbar.set)
        output_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=output_text.yview)

        def log_step(text: str) -> None:
            output_text.config(state=tk.NORMAL)
            output_text.insert(tk.END, text + "\n")
            output_text.see(tk.END)
            output_text.update()
            output_text.config(state=tk.DISABLED)

        def show_steps() -> None:
            output_text.config(state=tk.NORMAL)
            output_text.delete(1.0, tk.END)
            output_text.config(state=tk.DISABLED)

            n = self.graph.n
            log_step("=" * 70)
            log_step("ПРОЦЕДУРА ПРЕОБРАЗОВАНИЯ ГРАФА")
            log_step("=" * 70)
            log_step("")

            # Шаг 1: Список смежности
            log_step("ШАГ 1: ИСХОДНОЕ ПРЕДСТАВЛЕНИЕ - СПИСОК СМЕЖНОСТИ")
            log_step("-" * 70)
            log_step("Внутри программа хранит граф как список смежности (множества).\n")
            adj_list = self.graph.to_adj_list()
            for i, neighbors in enumerate(adj_list, start=1):
                if neighbors:
                    log_step(f"  Вершина {i}: {neighbors}")
                else:
                    log_step(f"  Вершина {i}: (нет соседей)")
            log_step("")
            time.sleep(0.5)

            # Шаг 2: Матрица смежности
            log_step("ШАГ 2: ПРЕОБРАЗОВАНИЕ В МАТРИЦУ СМЕЖНОСТИ")
            log_step("-" * 70)
            log_step("Для каждой пары вершин (i, j) в матрице M[i][j]:")
            log_step("  - если есть ребро между i и j, то M[i][j] = 1")
            log_step("  - иначе M[i][j] = 0\n")

            mat = self.graph.to_adj_matrix()
            width = 3
            header = "    " + " ".join(f"{j+1:>{width}}" for j in range(len(mat[0])))
            log_step(header)
            for i, row in enumerate(mat, start=1):
                log_step(f"{i:>{3}} " + " ".join(f"{x:>{width}}" for x in row))
            log_step("")
            time.sleep(0.5)

            # Шаг 3: Матрица инцидентности
            log_step("ШАГ 3: ПРЕОБРАЗОВАНИЕ В МАТРИЦУ ИНЦИДЕНТНОСТИ")
            log_step("-" * 70)
            log_step("Матрица V x E (вершины x рёбра).")
            log_step("Для каждого ребра e и вершины v:")
            if self.graph.directed:
                log_step("  - M[v][e] = -1, если ребро из v")
                log_step("  - M[v][e] = +1, если ребро в v")
            else:
                log_step("  - M[v][e] = 1, если v инцидентна ребру")
                log_step("  - M[v][e] = 2, если ребро - петля (v-v)")
            log_step("  - M[v][e] = 0, иначе\n")

            edges = self.graph.to_edge_list()
            incidence = self.graph.to_incidence()

            log_step(f"Найденные рёбра: {len(edges)}")
            for idx, (u, v) in enumerate(edges, start=1):
                log_step(f"  Ребро {idx}: ({u+1}, {v+1})")
            log_step("")

            if incidence:
                width = 3
                header = "    " + " ".join(f"{e+1:>{width}}" for e in range(len(edges)))
                log_step(header)
                for i, row in enumerate(incidence, start=1):
                    log_step(f"{i:>{3}} " + " ".join(f"{x:>{width}}" for x in row))
            log_step("")
            time.sleep(0.5)

            # Шаг 4: Список рёбер
            log_step("ШАГ 4: ПРЕДСТАВЛЕНИЕ СПИСКОМ РЁБЕР")
            log_step("-" * 70)
            log_step("Простой перечень всех рёбер в графе:\n")
            for idx, (u, v) in enumerate(edges, start=1):
                log_step(f"  Ребро {idx}: ({u+1}, {v+1})")
            log_step("")
            time.sleep(0.5)

            # Итоговая информация
            log_step("=" * 70)
            log_step("ИТОГОВАЯ ИНФОРМАЦИЯ")
            log_step("=" * 70)
            log_step(f"Тип графа: {'Ориентированный' if self.graph.directed else 'Неориентированный'}")
            log_step(f"Количество вершин: {n}")
            log_step(f"Количество рёбер: {len(edges)}")
            log_step(f"Плотность графа: {len(edges) / (n * (n-1) / 2) if n > 1 else 0:.2%}")
            log_step("=" * 70)

        # Запускаем показ шагов в отдельном потоке
        thread = threading.Thread(target=show_steps, daemon=True)
        thread.start()



if __name__ == "__main__":
    root = tk.Tk()
    root.title("Лабораторная: Хранение графа в памяти")
    root.geometry("900x700")
    root.resizable(True, True)
    
    app = GraphGUI(root)
    root.mainloop()