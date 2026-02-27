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
    
    def traverse(self, start_vertex: int) -> List[int]:
        """Выполнить DFS обход с заданной вершины"""
        # Проверяем корректность стартовой вершины
        if start_vertex < 0 or start_vertex >= self.graph.n:
            print(f"Ошибка: неверная стартовая вершина {start_vertex}!")
            return []
        
        # Сбрасываем состояние
        self.visited = [False] * self.graph.n
        self.result = []
        
        # Выполняем DFS
        self.dfs(start_vertex)
        
        return self.result
    
    def print_result(self, vertices: List[int]) -> None:
        """Вывести результат обхода в виде списка"""
        if not vertices:
            print("Список вершин: (граф пуст или вершина недоступна)")
            return
        
        print("Обход графа в глубину (DFS):")
        print(f"Список вершин: {' -> '.join(map(str, vertices))}")


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


def main():
    print("=" * 50)
    print("ЛАБОРАТОРНАЯ РАБОТА: ОБХОД ГРАФА В ГЛУБИНУ (DFS)")
    print("=" * 50)
    
    while True:
        # Выбор режима
        print("\n" + "=" * 50)
        print("ГЛАВНОЕ МЕНЮ")
        print("=" * 50)
        print("1 - Создать собственный граф")
        print("2 - Использовать пример графа")
        print("3 - Выход")
        
        choice = input("\nВаш выбор (1, 2 или 3): ").strip()
        
        if choice == "1":
            graph = create_graph_interactive()
            if graph is None:
                continue
        elif choice == "2":
            # Пример графа
            graph = Graph(directed=False)
            graph.add_edge(0, 1)
            graph.add_edge(0, 2)
            graph.add_edge(1, 3)
            graph.add_edge(2, 3)
            graph.add_edge(2, 4)
            graph.add_edge(3, 5)
            graph.add_edge(4, 5)
            print("\n✓ Загружен пример графа")
        elif choice == "3":
            print("\n" + "=" * 50)
            print("Спасибо за использование программы!")
            print("=" * 50)
            break
        else:
            print("❌ Пожалуйста, выберите 1, 2 или 3")
            continue
        
        # Показываем граф
        print("\n" + "-" * 50)
        print("ИНФОРМАЦИЯ О ГРАФЕ")
        print("-" * 50)
        graph_type = "Ориентированный" if graph.directed else "Неориентированный"
        print(f"Тип графа: {graph_type}")
        print(f"Всего вершин в графе: {graph.n}")
        print()
        graph.print_adjacency_list()
        
        # Выполняем DFS обход
        print("\n" + "-" * 50)
        print("ОБХОД ГРАФА В ГЛУБИНУ")
        print("-" * 50)
        
        while True:
            try:
                start_vertex = int(input(f"\nВведите стартовую вершину (0-{graph.n-1}) или -1 для выхода в меню: ").strip())
                
                if start_vertex == -1:
                    break
                
                if start_vertex < 0 or start_vertex >= graph.n:
                    print(f"❌ Ошибка: вершина должна быть в диапазоне 0-{graph.n-1}")
                    continue
                
                dfs_traversal = DFSTraversal(graph)
                result = dfs_traversal.traverse(start_vertex)
                print()
                dfs_traversal.print_result(result)
                
            except ValueError:
                print("❌ Пожалуйста, введите корректное число")
            except KeyboardInterrupt:
                print("\n")
                break


if __name__ == "__main__":
    main()
