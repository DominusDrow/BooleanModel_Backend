import re

import nltk
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from flask_cors import CORS  # Importa Flask-CORS
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

app = Flask(__name__)
CORS(app)

# Lista de URLs de las páginas web
urls = [
    "https://es.wikipedia.org/wiki/Inteligencia_artificial",
    "https://es.wikipedia.org/wiki/Prueba_de_Turing",
    "https://es.wikipedia.org/wiki/Aut%C3%B3mata_(mec%C3%A1nico)",
    "https://es.wikipedia.org/wiki/Robot_aut%C3%B3nomo",
    "https://es.wikipedia.org/wiki/John_von_Neumann",
    "https://es.wikipedia.org/wiki/Computaci%C3%B3n_cu%C3%A1ntica",
    "https://es.wikipedia.org/wiki/Programaci%C3%B3n",
    "https://es.wikipedia.org/wiki/Python",
    "https://es.wikipedia.org/wiki/GNU/Linux",
    "https://es.wikipedia.org/wiki/Linus_Torvalds",
    "https://es.wikipedia.org/wiki/Ciencias_de_la_computaci%C3%B3n",
    "https://es.wikipedia.org/wiki/Charles_Babbage",
    "https://es.wikipedia.org/wiki/Ada_Lovelace",
    "https://es.wikipedia.org/wiki/Algoritmo",
]

# Crear un diccionario para almacenar los términos por URL
document_terms = {}

# Iterar a través de las URLs y extraer los términos
for url in urls:
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        headings = soup.find_all(["h1", "h2"])
        extracted_text = [p.get_text() for p in paragraphs] + [
            h.get_text() for h in headings
        ]
        # Aplicar el preprocesamiento de términos (eliminación de stopwords, stemming, etc.) aquí si es necesario
        document_terms[url] = extracted_text


nltk.download("punkt")

# Recorrer los términos de cada documento en document_terms
for document_url, terms in document_terms.items():
    tokenized_terms = []  # Lista para almacenar los términos tokenizados
    for term in terms:
        term = re.sub(r"[^\w\s]", "", term)
        term = re.sub(r"\d", "", term)
        if term:
            tokens = word_tokenize(
                term.lower()
            )  # Convertir a minúsculas para la normalización
            tokenized_terms.extend(tokens)  # Agregar tokens a la lista

    document_terms[
        document_url
    ] = tokenized_terms  # Reemplazar la lista de términos por los tokens

nltk.download("stopwords")

# Obtener una lista de palabras vacías en español
stop_words = set(stopwords.words("spanish"))

# Recorrer y reemplazar los términos de cada documento en document_terms
for document_url, terms in document_terms.items():
    # Filtrar términos que no son palabras vacías
    filtered_terms = [term for term in terms if term not in stop_words]
    document_terms[
        document_url
    ] = filtered_terms  # Reemplazar la lista de términos por los términos filtrados

# Crear un objeto SnowballStemmer para español
stemmer = SnowballStemmer("spanish")

# Recorrer y reemplazar los términos de cada documento en document_terms con stemming
for document_url, terms in document_terms.items():
    # Aplicar stemming a los términos
    stemmed_terms = [stemmer.stem(term) for term in terms]
    document_terms[
        document_url
    ] = stemmed_terms  # Reemplazar la lista de términos por los términos con stemming


class BTreeNode:
    def __init__(self):
        self.keys = []  # Lista de términos
        self.links = {}  # Diccionario de términos a lista de enlaces
        self.children = []  # Lista de nodos hijos


class BTree:
    def __init__(self):
        self.root = BTreeNode()
        self.degree = 5  # Grado máximo del árbol B

    def insert(self, term, link):
        self._insert_recursive(self.root, term, link)

    def _insert_recursive(self, node, term, link):
        # Si el nodo es una hoja, inserta el término y el enlace en la posición adecuada
        if not node.children:
            index = 0
            while index < len(node.keys) and term > node.keys[index]:
                index += 1
            if index < len(node.keys) and term == node.keys[index]:
                # Si el término ya existe, agrega el enlace a la lista existente
                if link not in node.links[term]:
                    node.links[term].append(link)
            else:
                # Si el término no existe, inserta el término y crea una nueva lista de enlaces
                node.keys.insert(index, term)
                node.links[term] = [link]
        else:
            # Encuentra el hijo adecuado
            index = 0
            while index < len(node.keys) and term > node.keys[index]:
                index += 1
            # Inserta recursivamente en el hijo correspondiente
            self._insert_recursive(node.children[index], term, link)
            # Divide el nodo si es necesario
            if len(node.keys) == self.degree - 1:
                self._split_node(node)

    def _split_node(self, node):
        # Divide un nodo que ha alcanzado su capacidad máxima
        middle_index = len(node.keys) // 2
        middle_term = node.keys[middle_index]
        left_child = BTreeNode()
        right_child = BTreeNode()
        left_child.keys = node.keys[:middle_index]
        left_child.links = {term: node.links[term] for term in node.keys[:middle_index]}
        right_child.keys = node.keys[middle_index + 1 :]
        right_child.links = {
            term: node.links[term] for term in node.keys[middle_index + 1 :]
        }
        if node.children:
            left_child.children = node.children[: middle_index + 1]
            right_child.children = node.children[middle_index + 1 :]
        node.keys = [middle_term]
        node.links = {middle_term: node.links[middle_term]}
        node.children = [left_child, right_child]

    def search(self, term):
        return self._search_recursive(self.root, term)

    def _search_recursive(self, node, term):
        index = 0
        while index < len(node.keys) and term > node.keys[index]:
            index += 1
        if index < len(node.keys) and term == node.keys[index]:
            return node.links[term]
        elif node.children:
            return self._search_recursive(node.children[index], term)
        else:
            return None


# Uso
tree = BTree()

# Insertar términos y enlaces del diccionario
for link, terms in document_terms.items():
    for term in terms:
        tree.insert(term, link)


# Función para convertir una consulta en lenguaje natural en notación posfija
def natural_to_postfix(query):
    # Eliminar signos de puntuación utilizando una expresión regular
    query = re.sub(r"[^\w\s]", "", query)

    # Tokenizar la consulta en lenguaje natural
    tokens = nltk.word_tokenize(query.lower())

    # Crear una pila para convertir a notación posfija
    output = []
    operator_stack = []

    # Diccionario para prioridades de operadores
    precedence = {"o": 1, "y": 2, "no": 3}

    for token in tokens:
        if token in ("y", "o", "no"):
            while (
                operator_stack
                and operator_stack[-1] != "("
                and precedence[token] <= precedence.get(operator_stack[-1], 0)
            ):
                output.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == "(":
            operator_stack.append(token)
        elif token == ")":
            while operator_stack and operator_stack[-1] != "(":
                output.append(operator_stack.pop())
            if operator_stack and operator_stack[-1] == "(":
                operator_stack.pop()
        elif token not in stop_words:  # Evitar eliminar "y", "o" y "no"
            output.append(stemmer.stem(token))

    while operator_stack:
        output.append(operator_stack.pop())

    return output


# Función para buscar en el árbol B según la consulta en notación posfija
def search_in_btree(btree, postfix_query):
    stack = []

    for token in postfix_query:
        if token in ("y", "o", "no"):
            if token == "y":
                operand2 = stack.pop()
                operand1 = stack.pop()
                if operand1 is None or operand2 is None:
                    stack.append(None)
                else:
                    result = set(operand1) & set(operand2)
                    stack.append(result)
            elif token == "o":
                operand2 = stack.pop()
                operand1 = stack.pop()
                if operand1 is None:
                    stack.append(operand2)
                elif operand2 is None:
                    stack.append(operand1)
                else:
                    result = set(operand1) | set(operand2)
                    stack.append(result)
            elif token == "no":
                operand = stack.pop()
                if operand is None:
                    stack.append(None)
                else:
                    all_terms = set(
                        btree.root.keys
                    )  # Obtener todos los términos del árbol B
                    result = all_terms - set(operand)
                    stack.append(result)
        else:
            stack.append(btree.search(token))  # Realizar la búsqueda en el árbol B

    if stack:
        return stack[0]
    else:
        return []


@app.route("/boolean-query", methods=["POST"])
def boolean_query():
    try:
        query = request.json.get("query")  # Recibe la consulta desde el cuerpo
        postfix_query = natural_to_postfix(query)
        results = search_in_btree(tree, postfix_query)
        if results:
            return jsonify({"results": list(results)})
        else:
            return jsonify({"results": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 400  # Bad request


# Ejecuta la aplicación Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
