#include <iostream>     // Para imprimir en pantalla o mostrar errores
#include <vector>       // Para manejar listas (vectores) de datos
#include <string>       // Para trabajar con texto
#include <sstream>      // Para dividir strings fácilmente
#include <fstream>      // Para leer y escribir archivos
#include <climits>      // Para usar INT_MAX (valor más alto posible de int)
#include <omp.h>        // Permite hacer tareas en paralelo (usando múltiples núcleos)
#include <algorithm>    // Para usar funciones como remove (quitar caracteres)

// Usamos espacio de nombres std para no escribir std:: en todo
using namespace std;

// Esta función toma un string que representa una matriz (como [[0,1,2],[3,4,5]])
// y lo convierte a una matriz de enteros que se puede usar en el programa
vector<vector<int>> parseMatrix(const string& input) {
    vector<vector<int>> matrix;
    string cleaned = input;

    // Quitamos los corchetes para dejar solo los números
    cleaned.erase(remove(cleaned.begin(), cleaned.end(), '['), cleaned.end());
    cleaned.erase(remove(cleaned.begin(), cleaned.end(), ']'), cleaned.end());

    stringstream ss(cleaned);
    string token;
    vector<int> row;

    // Calculamos cuántas filas hay mirando los corchetes originales
    int dimension = 0;
    for (char c : input) {
        if (c == '[') dimension++;
    }

    int count = 0;

    // Vamos separando los números por comas y armamos las filas
    while (getline(ss, token, ',')) {
        int value = stoi(token);
        row.push_back(value);
        count++;

        // Cuando completamos una fila, la agregamos a la matriz
        if (count % dimension == 0) {
            matrix.push_back(row);
            row.clear();
        }
    }

    return matrix;
}

// Esta función aplica el algoritmo de Dijkstra (para encontrar caminos más cortos)
// Se hace de forma paralela con OpenMP para que sea más rápido
vector<int> dijkstra(const vector<vector<int>>& graph, int source) {
    int n = graph.size();
    vector<int> dist(n, INT_MAX);     // Inicializamos todas las distancias como "infinito"
    vector<bool> visited(n, false);   // Marcamos que ningún nodo ha sido visitado

    dist[source] = 0;  // La distancia al nodo origen es cero

    // Repetimos esto (n - 1) veces como indica el algoritmo
    for (int i = 0; i < n - 1; ++i) {
        int minDist = INT_MAX;
        int minIndex = -1;

        // Buscamos el nodo no visitado con la distancia más corta (en paralelo)
        #pragma omp parallel for
        for (int v = 0; v < n; ++v) {
            if (!visited[v]) {
                #pragma omp critical
                {
                    if (dist[v] < minDist) {
                        minDist = dist[v];
                        minIndex = v;
                    }
                }
            }
        }

        if (minIndex == -1) break; // Si no encontramos nodo, salimos
        visited[minIndex] = true;

        // Actualizamos las distancias de los vecinos (también en paralelo)
        #pragma omp parallel for
        for (int v = 0; v < n; ++v) {
            if (!visited[v] && graph[minIndex][v] > 0 && dist[minIndex] != INT_MAX) {
                int newDist = dist[minIndex] + graph[minIndex][v];
                if (newDist < dist[v]) {
                    dist[v] = newDist;
                }
            }
        }
    }

    return dist;
}

// Esta es la función principal. Recibe tres argumentos por consola:
// 1. La matriz en formato string
// 2. El nodo de inicio
// 3. La ruta del archivo donde se guardarán los resultados
int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Uso: programa \"matriz\" \"origen\" \"archivo_salida\"\n";
        return EXIT_FAILURE;
    }

    string matrixInput = argv[1];
    int startVertex = stoi(argv[2]);
    string outputPath = argv[3];

    vector<vector<int>> graph = parseMatrix(matrixInput);

    // Si el nodo de inicio no existe, salimos con error
    if (startVertex < 0 || startVertex >= static_cast<int>(graph.size())) {
        cerr << "Error: vértice de inicio fuera de rango.\n";
        return EXIT_FAILURE;
    }

    // Ejecutamos el algoritmo
    vector<int> distances = dijkstra(graph, startVertex);

    // Guardamos los resultados en el archivo especificado
    ofstream outputFile(outputPath);
    if (!outputFile.is_open()) {
        cerr << "Error: no se pudo abrir el archivo de salida.\n";
        return EXIT_FAILURE;
    }

    outputFile << "Vértice\tDistancia desde el origen\n";
    for (size_t i = 0; i < distances.size(); ++i) {
        outputFile << i << "\t";
        if (distances[i] == INT_MAX) {
            outputFile << "Inalcanzable\n";
        } else {
            outputFile << distances[i] << "\n";
        }
    }
}

