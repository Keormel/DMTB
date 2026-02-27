from typing import List, Set, Dict
from dataclasses import dataclass, field

@dataclass
class Graph:
    """Класс представления графа со списком смежности"""
    directed: bool = False
    adj: List[Set[int]] = field(default_factory=list)
    
    @property
    def n(self) -> int:
        """Количество вершин в графе"""
        return len(self.adj)
    
    def add_vertex(self, vertex: int = None) -> None:
        """Добавить вершину в граф"""
        if vertex is None:
            self.adj.append(set())
        elif vertex >= len(self.adj):
            # Расширяем граф до нужного размера
            while len(self.adj) <= vertex:
                self.adj.append(set())
    
    def add_edge(self, from_vertex: int, to_vertex: int) -> None:
        """Добавить ребро в граф"""
        # Убедимся, что вершины существуют
        while len(self.adj) <= max(from_vertex, to_vertex):
            self.adj.append(set())
        
        # Добавляем ребро
        self.adj[from_vertex].add(to_vertex)
        
        # Если граф неориентированный, добавляем обратное ребро
        if not self.directed:
            self.adj[to_vertex].add(from_vertex)
    
    def print_adjacency_list(self) -> None:
        """Вывести список смежности"""
        print("Список смежности графа:")
        for vertex in range(self.n):
            neighbors = self.adj[vertex]
            if neighbors:
                print(f"Вершина {vertex}: {sorted(neighbors)}")
            else:
                print(f"Вершина {vertex}: (нет соседей)")


class DFSTraversal:
    """Класс для выполнения обхода графа в глубину"""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.visited = [False] * graph.n
        self.result = []
    
    def dfs(self, vertex: int) -> None:
        """Рекурсивный обход в глубину"""
        self.visited[vertex] = True
        self.result.append(vertex)
        
        # Посещаем всех соседей вершины
        for neighbor in sorted(self.graph.adj[vertex]):
            if not self.visited[neighbor]:
                self.dfs(neighbor)
    
    def traverse(self, start_vertex: int, visit_all: bool = True) -> List[int]:
        """Выполнить DFS обход с заданной вершины
        
        Args:
            start_vertex: начальная вершина
            visit_all: если True, посетить все компоненты связности; если False, только компоненту начальной вершины
        """
        # Проверяем корректность стартовой вершины
        if start_vertex < 0 or start_vertex >= self.graph.n:
            print(f"Ошибка: неверная стартовая вершина {start_vertex}!")
            return []
        
        # Сбрасываем состояние
        self.visited = [False] * self.graph.n
        self.result = []
        
        # Выполняем DFS с начальной вершины
        self.dfs(start_vertex)
        
        # Если нужно посетить все вершины, обходим оставшиеся компоненты
        if visit_all:
            for vertex in range(self.graph.n):
                if not self.visited[vertex]:
                    self.dfs(vertex)
        
        return self.result
    
    def print_result(self, vertices: List[int], show_components: bool = False) -> None:
        """Вывести результат обхода в виде списка"""
        if not vertices:
            print("Список вершин: (граф пуст или вершина недоступна)")
            return
        
        print("Обход графа в глубину (DFS):")
        print(f"Список вершин: {' -> '.join(map(str, vertices))}")
        print(f"Всего посещено вершин: {len(vertices)}")


class BFSTraversal:
    """Класс для выполнения обхода графа в ширину"""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.visited = [False] * graph.n
        self.result = []
    
    def bfs_component(self, start_vertex: int) -> None:
        """Обход в ширину одной компоненты связности"""
        queue = [start_vertex]
        self.visited[start_vertex] = True
        
        while queue:
            vertex = queue.pop(0)
            self.result.append(vertex)
            
            for neighbor in sorted(self.graph.adj[vertex]):
                if not self.visited[neighbor]:
                    self.visited[neighbor] = True
                    queue.append(neighbor)
    
    def traverse(self, start_vertex: int, visit_all: bool = True) -> List[int]:
        """Выполнить BFS обход с заданной вершины
        
        Args:
            start_vertex: начальная вершина
            visit_all: если True, посетить все компоненты связности; если False, только компоненту начальной вершины
        """
        if start_vertex < 0 or start_vertex >= self.graph.n:
            print(f"Ошибка: неверная стартовая вершина {start_vertex}!")
            return []
        
        self.visited = [False] * self.graph.n
        self.result = []
        
        # Обходим начальную компоненту
        self.bfs_component(start_vertex)
        
        # Если нужно посетить все вершины, обходим оставшиеся компоненты
        if visit_all:
            for vertex in range(self.graph.n):
                if not self.visited[vertex]:
                    self.bfs_component(vertex)
        
        return self.result
    
    def print_result(self, vertices: List[int]) -> None:
        """Вывести результат обхода в виде списка"""
        if not vertices:
            print("Список вершин: (граф пуст или вершина недоступна)")
            return
        
        print("Обход графа в ширину (BFS):")
        print(f"Список вершин: {' -> '.join(map(str, vertices))}")
        print(f"Всего посещено вершин: {len(vertices)}")


def create_graph_interactive() -> Graph:
    """Интерактивное создание графа"""
    print("\n" + "=" * 50)
    print("СОЗДАНИЕ ГРАФА")
    print("=" * 50)
    
    # Выбор типа графа
    while True:
        try:
            choice = input("\nВыберите тип графа:\n1 - Неориентированный\n2 - Ориентированный\nВаш выбор (1 или 2): ").strip()
            if choice == "1":
                directed = False
                print("✓ Выбран неориентированный граф")
                break
            elif choice == "2":
                directed = True
                print("✓ Выбран ориентированный граф")
                break
            else:
                print("❌ Пожалуйста, введите 1 или 2")
        except KeyboardInterrupt:
            print("\n❌ Отмена операции")
            return None
    
    # Количество вершин
    while True:
        try:
            num_vertices = int(input("\nВведите количество вершин (от 1 до 20): ").strip())
            if 1 <= num_vertices <= 20:
                print(f"✓ Граф будет иметь {num_vertices} вершин (0-{num_vertices-1})")
                break
            else:
                print("❌ Пожалуйста, введите число от 1 до 20")
        except ValueError:
            print("❌ Пожалуйста, введите корректное число")
    
    # Создаём граф
    graph = Graph(directed=directed)
    for i in range(num_vertices):
        graph.add_vertex()
    
    # Добавление рёбер
    print("\n" + "-" * 50)
    print("ДОБАВЛЕНИЕ РЁБЕР")
    print("-" * 50)
    print("Введите рёбра в формате: 'вершина1 вершина2'")
    print("Для завершения введите 'готово' или пустую строку\n")
    
    edges_added = 0
    while True:
        try:
            edge_input = input(f"Ребро {edges_added + 1}: ").strip()
            
            if edge_input.lower() in ['готово', 'done', '']:
                if edges_added == 0:
                    print("⚠️  Добавлено 0 рёбер. Граф с изолированными вершинами.")
                else:
                    print(f"✓ Добавлено {edges_added} рёбер")
                break
            
            parts = edge_input.split()
            if len(parts) != 2:
                print("❌ Ошибка: введите два номера вершин через пробел")
                continue
            
            try:
                from_vertex = int(parts[0])
                to_vertex = int(parts[1])
                
                if from_vertex < 0 or from_vertex >= num_vertices or to_vertex < 0 or to_vertex >= num_vertices:
                    print(f"❌ Ошибка: вершины должны быть в диапазоне 0-{num_vertices-1}")
                    continue
                
                if from_vertex == to_vertex:
                    print("⚠️  Петля (ребро из вершины в саму себя)")
                
                graph.add_edge(from_vertex, to_vertex)
                print(f"✓ Добавлено ребро {from_vertex} -> {to_vertex}")
                edges_added += 1
                
            except ValueError:
                print("❌ Ошибка: введите целые числа")
        
        except KeyboardInterrupt:
            print("\n❌ Отмена операции")
            return None
    
    return graph


def create_sample_graph() -> Graph:
    """Создать пример графа для демонстрации"""
    graph = Graph(directed=False)
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    graph.add_edge(2, 4)
    graph.add_edge(3, 5)
    graph.add_edge(4, 5)
    return graph


def create_directed_sample() -> Graph:
    """Создать пример ориентированного графа"""
    graph = Graph(directed=True)
    graph.add_vertex(0)
    graph.add_vertex(1)
    graph.add_vertex(2)
    graph.add_vertex(3)
    graph.add_vertex(4)
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    return graph


def find_connected_components(graph: Graph) -> List[List[int]]:
    """Найти все компоненты связности в графе"""
    visited = [False] * graph.n
    components = []
    
    def dfs_component(vertex: int, component: List[int]) -> None:
        visited[vertex] = True
        component.append(vertex)
        for neighbor in sorted(graph.adj[vertex]):
            if not visited[neighbor]:
                dfs_component(neighbor, component)
    
    for vertex in range(graph.n):
        if not visited[vertex]:
            component = []
            dfs_component(vertex, component)
            components.append(sorted(component))
    
    return components


def print_connected_components(graph: Graph) -> None:
    """Вывести все компоненты связности"""
    components = find_connected_components(graph)
    
    if len(components) == 1:
        print(f"✓ Граф связный (1 компонента, {len(components[0])} вершин)")
    else:
        print(f"✓ Граф несвязный ({len(components)} компоненты связности):")
        for i, component in enumerate(components, 1):
            print(f"   Компонента {i}: {component} ({len(component)} вершин)")


def main():
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА: ОБХОД ГРАФА (DFS и BFS)")
    print("=" * 60)
    
    while True:
        # Выбор режима
        print("\n" + "=" * 60)
        print("ГЛАВНОЕ МЕНЮ")
        print("=" * 60)
        print("1 - Создать собственный граф")
        print("2 - Использовать пример неориентированного графа")
        print("3 - Использовать пример ориентированного графа")
        print("4 - Выход")
        
        choice = input("\nВаш выбор (1, 2, 3 или 4): ").strip()
        
        if choice == "1":
            graph = create_graph_interactive()
            if graph is None:
                continue
        elif choice == "2":
            graph = create_sample_graph()
            print("\n✓ Загружен пример неориентированного графа")
        elif choice == "3":
            graph = create_directed_sample()
            print("\n✓ Загружен пример ориентированного графа")
        elif choice == "4":
            print("\n" + "=" * 60)
            print("Спасибо за использование программы!")
            print("=" * 60)
            break
        else:
            print("❌ Пожалуйста, выберите 1, 2, 3 или 4")
            continue
        
        # Показываем граф
        print("\n" + "-" * 60)
        print("ИНФОРМАЦИЯ О ГРАФЕ")
        print("-" * 60)
        graph_type = "Ориентированный" if graph.directed else "Неориентированный"
        print(f"Тип графа: {graph_type}")
        print(f"Всего вершин в графе: {graph.n}")
        print()
        graph.print_adjacency_list()
        
        # Показываем компоненты связности
        print("\n" + "-" * 60)
        print("АНАЛИЗ СВЯЗНОСТИ")
        print("-" * 60)
        print_connected_components(graph)
        
        # Выполняем обход
        print("\n" + "-" * 60)
        print("ОБХОД ГРАФА")
        print("-" * 60)
        
        while True:
            try:
                print("\n1 - DFS (обход в глубину)")
                print("2 - BFS (обход в ширину)")
                print("3 - Вернуться в главное меню")
                
                algo_choice = input("\nВыберите алгоритм (1, 2 или 3): ").strip()
                
                if algo_choice == "3":
                    break
                elif algo_choice not in ["1", "2"]:
                    print("❌ Пожалуйста, выберите 1, 2 или 3")
                    continue
                
                start_vertex = int(input(f"\nВведите стартовую вершину (0-{graph.n-1}): ").strip())
                
                if start_vertex < 0 or start_vertex >= graph.n:
                    print(f"❌ Ошибка: вершина должна быть в диапазоне 0-{graph.n-1}")
                    continue
                
                print("\nРежимы обхода:")
                print("1 - Только компонента начальной вершины")
                print("2 - ВСЕ вершины графа (все компоненты)")
                mode_choice = input("Выберите режим (1 или 2) [по умолчанию 2]: ").strip() or "2"
                
                visit_all = mode_choice != "1"
                
                if algo_choice == "1":
                    dfs_traversal = DFSTraversal(graph)
                    result = dfs_traversal.traverse(start_vertex, visit_all=visit_all)
                    print()
                    dfs_traversal.print_result(result)
                else:
                    bfs_traversal = BFSTraversal(graph)
                    result = bfs_traversal.traverse(start_vertex, visit_all=visit_all)
                    print()
                    bfs_traversal.print_result(result)
                
            except ValueError:
                print("❌ Пожалуйста, введите корректное число")
            except KeyboardInterrupt:
                print("\n")
                break


if __name__ == "__main__":
    main()
