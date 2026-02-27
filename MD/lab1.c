#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int n;
    int **adj;
} Graph;

int** allocMatrix(int n, int m) {
    int **a = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        a[i] = calloc(m, sizeof(int));
    }
    return a;
}

Graph createGraph(int n) {
    Graph g;
    g.n = n;
    g.adj = allocMatrix(n, n);
    return g;
}


void freeMatrix(int **a, int n) {
    for (int i = 0; i < n; i++)
        free(a[i]);
    free(a);
}

void freeGraph(Graph *g) {
    freeMatrix(g->adj, g->n);
    g->adj = NULL;
    g->n = 0;
}



// ==== ВВОД ГРАФА ====


// ввод списка смежности
Graph readAdjList() {
    int n;
    printf("Количество вершин: ");
    scanf("%d", &n);

    Graph g = createGraph(n);

    for (int i = 0; i < n; i++) {
        printf("Введите соседей вершины %d: ", i+1);
        int v = -1;
        while (v != 0) {
            scanf("%d", &v);
            if (v == 0) break;
            if (v >= 1 && v <= n)
                g.adj[i][v-1] = 1;
            else
                printf("Неверная вершина! Введите снова: ");
        }
    }
    return g;
}

// ввод матрицы смежности
Graph readAdjMatrix() {
    int n;
    printf("Количество вершин: ");
    scanf("%d", &n);

    Graph g = createGraph(n);

    printf("Введите матрицу смежности:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &g.adj[i][j]);

    return g;
}

// ввод матрицы инцидентности
Graph readIncMatrix() {
    int n, m;
    printf("Количество вершин: ");
    scanf("%d", &n);
    printf("Количество рёбер: ");
    scanf("%d", &m);

    int **inc = allocMatrix(n, m);

    printf("Введите матрицу инцидентности (-1 начало, 1 конец):\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            scanf("%d", &inc[i][j]);

    Graph g = createGraph(n);

    for (int j = 0; j < m; j++) {
        int from = -1, to = -1;
        for (int i = 0; i < n; i++) {
            if (inc[i][j] == -1) from = i;
            if (inc[i][j] == 1)  to = i;
        }
        if (from != -1 && to != -1)
            g.adj[from][to] = 1;
    }

    freeMatrix(inc, n);
    return g;
}


// ==== ВЫВОД ГРАФА ====


// вывод списка смежности
void printAdjList(Graph *g) {
    printf("\nСписок смежности:\n");
    for (int i = 0; i < g->n; i++) {
        printf("%d: ", i+1);
        for (int j = 0; j < g->n; j++)
            if (g->adj[i][j])
                printf("%d ", j+1);
        printf("\n");
    }
}

// вывод матрицы смежности
void printAdjMatrix(Graph *g) {
    printf("\nМатрица смежности:\n\n");

    printf("    ");
    for (int j = 0; j < g->n; j++)
        printf("%3d", j+1);
    printf("\n");

    for (int i = 0; i < g->n; i++) {
        printf("%3d ", i+1);

        for (int j = 0; j < g->n; j++)
            printf("%3d", g->adj[i][j]);

        printf("\n");
    }
}


// вывод матрицы инцидентности
void printIncMatrix(Graph *g) {
    int edges = 0;

    // считаем рёбра
    for (int i = 0; i < g->n; i++)
        for (int j = 0; j < g->n; j++)
            if (g->adj[i][j])
                edges++;

    int **inc = allocMatrix(g->n, edges);

    int e = 0;
    for (int i = 0; i < g->n; i++) {
        for (int j = 0; j < g->n; j++) {
            if (g->adj[i][j]) {
                if (i == j) {// петля
                    inc[i][e] = 2;
                }
                else {
                    inc[i][e] = -1;
                    inc[j][e] = 1;
                }
                e++;
            }
        }
    }


    printf("\nМатрица инцидентности:\n\n");

    // верхняя строка — номера рёбер
    printf("    ");
    for (int j = 0; j < edges; j++)
        printf("%3d", j+1);
    printf("\n");

    // строки
    for (int i = 0; i < g->n; i++) {
        printf("%3d ", i+1); // подпись вершины

        for (int j = 0; j < edges; j++)
            printf("%3d", inc[i][j]);

        printf("\n");
    }

    freeMatrix(inc, g->n);
}


// ==== РЕДАКТИРОВАНИЕ ГРАФА ====
void addVertex(Graph *g) {
    int newN = g->n + 1;

    int **newAdj = allocMatrix(newN, newN);

    for (int i = 0; i < g->n; i++)
        for (int j = 0; j < g->n; j++)
            newAdj[i][j] = g->adj[i][j];

    freeMatrix(g->adj, g->n);
    g->adj = newAdj;
    g->n = newN;

    printf("Вершина добавлена. Теперь вершин: %d\n", g->n);
}

void removeVertex(Graph *g, int v) {
    if (v < 0 || v >= g->n) {
        printf("Неверный номер вершины\n");
        return;
    }

    int newN = g->n - 1;
    int **newAdj = allocMatrix(newN, newN);

    int ni = 0, nj;
    for (int i = 0; i < g->n; i++) {
        if (i == v) continue;
        nj = 0;
        for (int j = 0; j < g->n; j++) {
            if (j == v) continue;
            newAdj[ni][nj++] = g->adj[i][j];
        }
        ni++;
    }

    freeMatrix(g->adj, g->n);
    g->adj = newAdj;
    g->n = newN;

    printf("Вершина удалена. Теперь вершин: %d\n", g->n);
}

void addEdge(Graph *g, int from, int to) {
    if (from<0||from>=g->n||to<0||to>=g->n){
        printf("Неверные вершины\n");
        return;
    }
    g->adj[from][to] = 1;
}

void removeEdge(Graph *g, int from, int to) {
    if (from<0||from>=g->n||to<0||to>=g->n){
        printf("Неверные вершины\n");
        return;
    }
    g->adj[from][to] = 0;
}


// ==== МЕНЮ ====


int main() {
    Graph g;
    int choice_in = -1;

    printf("\nСпособ ввода графа:\n");
    printf("    1 — список смежности\n");
    printf("    2 — матрица смежности\n");
    printf("    3 — матрица инцидентности\n");
    
    while (choice_in < 1 || choice_in > 3) {
        printf("Ваш выбор: ");
        scanf("%d", &choice_in);
        if (choice_in == 1) {
            g = readAdjList();
        }
        else if (choice_in == 2) {
            g = readAdjMatrix();
        } 
        else if (choice_in == 3) {
            g = readIncMatrix();
        }
        else {
            printf("Неверный выбор!\n");
        }
    }

    int choice_out = -1;
    while (choice_out != 0) {
        printf("\nКак вывести граф?\n");
        printf("1 — список смежности\n");
        printf("2 — матрица смежности\n");
        printf("3 — матрица инцидентности\n");
        printf("4 — корректировка\n");
        printf("0 — выход\n");

        printf("Ваш выбор: ");
        scanf("%d", &choice_out);
        if (choice_out == 1) {
            printAdjList(&g);
        }
        else if (choice_out == 2) {
            printAdjMatrix(&g);
        }
        else if (choice_out == 3) {
            printIncMatrix(&g);
        }
        else if (choice_out == 4) {
            int edit = -1;
            while (edit != 0) {
                printf("\nРедактирование:\n");
                printf("1 — добавить вершину\n");
                printf("2 — удалить вершину\n");
                printf("3 — добавить ребро\n");
                printf("4 — удалить ребро\n");
                printf("0 — назад\n");

                printf("Ваш выбор: ");
                scanf("%d", &edit);

                if (edit == 1) {
                    addVertex(&g);
                }
                else if (edit == 2) {
                    int v;
                    printf("Номер вершины: ");
                    scanf("%d", &v);
                    removeVertex(&g, v-1);
                }
                else if (edit == 3) {
                    int a,b;
                    printf("Откуда -> куда: ");
                    scanf("%d%d",&a,&b);
                    addEdge(&g, a-1, b-1);
                }
                else if (edit == 4) {
                    int a,b;
                    printf("Какое ребро удалить: ");
                    scanf("%d %d", &a, &b);
                    removeEdge(&g, a-1, b-1);
                }
                else if (edit == 0) {
                    printf("Возврат в меню...\n");
                }
                else {
                    printf("Неверный выбор!\n");
                }
            }
        }

        else if (choice_out == 0) {
            printf("Выход...\n");
        }
        else {
            printf("Неверный выбор!\n");
        }
    }
    freeGraph(&g);
    return 0;
}
