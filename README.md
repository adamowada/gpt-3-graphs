# gpt-3-graphs

Asking GPT-3 to implement different graph algorithms in python

implement breadth first traversal in a unweighted, undirected graph using an adjacency list

```python
class Graph:
    def __init__(self):
        self.vertices = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertices[key] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertices

    def addEdge(self, f, t, cost=0):
        if f not in self.vertices:
            nv = self.addVertex(f)
        if t not in self.vertices:
            nv = self.addVertex(t)
        self.vertices[f].addNeighbor(self.vertices[t], cost)

    def getVertices(self):
        return list(self.vertices.keys())

    def __iter__(self):
        return iter(self.vertices.values())

def bfs(g, start):
    start.setDistance(0)
    start.setPred(None)
    vertQueue = Queue()
    vertQueue.enqueue(start)
    while (vertQueue.size() > 0):
        currentVert = vertQueue.dequeue()
        for nbr in currentVert.getConnections():
            if (nbr.getColor() == 'white'):
                nbr.setColor('gray')
                nbr.setDistance(currentVert.getDistance() + 1)
                nbr.setPred(currentVert)
                vertQueue.enqueue(nbr)
        currentVert.setColor('black')
```


## Depth First Search

implement depth first traversal in a unweighted, undirected graph using an adjacency list

```python
def dfs(g, start):
    start.setDistance(0)
    start.setPred(None)
    vertQueue = Queue()
    vertQueue.enqueue(start)
    while (vertQueue.size() > 0):
        currentVert = vertQueue.dequeue()
        for nbr in currentVert.getConnections():
            if (nbr.getColor() == 'white'):
                nbr.setColor('gray')
                nbr.setDistance(currentVert.getDistance() + 1)
                nbr.setPred(currentVert)
                vertQueue.enqueue(nbr)
        currentVert.setColor('black')
```


## Topological Sort

implement topological sort in a unweighted, directed graph using an adjacency list

```python
def topologicalSort(g):
    topo = []
    for aVertex in g:
        aVertex.setColor('white')
        aVertex.setPred(-1)
    for aVertex in g:
        if aVertex.getColor() == 'white':
            dfsvisit(g, aVertex, topo)
    return topo

def dfsvisit(g, start, topo):
    start.setColor('gray')
    for next in start.getConnections():
        if next.getColor() == 'white':
            next.setPred(start)
            dfsvisit(g, next, topo)
    start.setColor('black')
    topo.insert(0, start)
```


## Dijkstra's Algorithm

implement Dijkstra's algorithm in a weighted, directed graph using an adjacency list

```python
def dijkstra(aGraph, start):
    pq = PriorityQueue()
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(),v) for v in aGraph])
    while not pq.isEmpty():
        currentVert = pq.delMin()
        for nextVert in currentVert.getConnections():
            newDist = currentVert.getDistance() \
                    + currentVert.getWeight(nextVert)
            if newDist < nextVert.getDistance():
                nextVert.setDistance( newDist )
                nextVert.setPred(currentVert)
                pq.decreaseKey(nextVert,newDist)
```


## Prim's Algorithm

implement Prim's algorithm in a weighted, undirected graph using an adjacency list

```python
def prim(G, start):
    pq = PriorityQueue()
    for v in G:
        v.setDistance(sys.maxsize)
        v.setPred(None)
    start.setDistance(0)
    pq.buildHeap([(v.getDistance(),v) for v in G])
    while not pq.isEmpty():
        currentVert = pq.delMin()
        for nextVert in currentVert.getConnections():
            newCost = currentVert.getWeight(nextVert)
            if nextVert in pq and newCost < nextVert.getDistance():
                nextVert.setPred(currentVert)
                nextVert.setDistance(newCost)
                pq.decreaseKey(nextVert,newCost)
```


## Kruskal's Algorithm

implement Kruskal's algorithm in a weighted, undirected graph using an adjacency list

```python
def Kruskal(G):
    forest = []
    for v in G:
        forest.append(Tree(v))
    edges = list(G.edges())
    edges.sort()
    for e in edges:
        weight, vert1, vert2 = e
        t1 = find(forest, vert1)
        t2 = find(forest, vert2)
        if t1 != t2:
            G.addEdge(vert1, vert2, weight)
            union(t1, t2)
    return G

def find(forest, vert):
    for tree in forest:
        if vert in tree:
            return tree
    return None

def union(t1, t2):
    for v in t2:
        t1.addVertex(v)
    forest.remove(t2)
```


## Bellman-Ford Algorithm

implement Bellman-Ford algorithm in a weighted, directed graph using an adjacency list

```python
def BellmanFord(G, start):
    for v in G:
        v.setDistance(sys.maxsize)
        v.setPred(None)
    start.setDistance(0)
    for i in range(0, len(G)-1):
        for u in G:
            for v in u.getConnections():
                newDist = u.getDistance() + u.getWeight(v)
                if newDist < v.getDistance():
                    v.setDistance(newDist)
                    v.setPred(u)
    for u in G:
        for v in u.getConnections():
            if v.getDistance() > u.getDistance() + u.getWeight(v):
                return False
    return True
```


## Floyd-Warshall Algorithm

implement Floyd-Warshall algorithm in a weighted, directed graph using an adjacency list

```python
def FloydWarshall(G):
    dist = {}
    for v in G:
        dist[v] = {}
        for w in G:
            dist[v][w] = v.getWeight(w)
            if v == w:
                dist[v][w] = 0
            elif dist[v][w] == 0:
                dist[v][w] = sys.maxsize
    for i in G:
        for j in G:
            for k in G:
                newDist = dist[j][i] + dist[i][k]
                if newDist < dist[j][k]:
                    dist[j][k] = newDist
    return dist
```


## A* Algorithm

implement A* algorithm in a weighted, directed graph using an adjacency list

```python
def aStarSearch(start, end, G):
    open = PriorityQueue()
    closed = {}
    open.insert(start, 0)
    while not open.isEmpty():
        current = open.pop()
        if current == end:
            return reconstructPath(current)
        closed[current] = True
        for neighbor in current.getConnections():
            if neighbor in closed:
                continue
            tentativeGScore = current.gScore + current.getWeight(neighbor)
            if neighbor not in open or tentativeGScore < neighbor.gScore:
                neighbor.cameFrom = current
                neighbor.gScore = tentativeGScore
                neighbor.fScore = neighbor.gScore + heuristic(neighbor, end)
                if neighbor not in open:
                    open.insert(neighbor, neighbor.fScore)
    return None

def reconstructPath(current):
    totalPath = [current]
    while current.cameFrom:
        current = current.cameFrom
        totalPath.append(current)
    totalPath.reverse()
    return totalPath

def heuristic(a, b):
    (x1, y1) = a.getPosition()
    (x2, y2) = b.getPosition()
    return abs(x1 - x2) + abs(y1 - y2)
```


## Trie

implement a trie

```python
class Trie:
    def __init__(self):
        self.root = {}
        self.endSymbol = "*"

    def add(self, word):
        current = self.root
        for letter in word:
            if letter not in current:
                current[letter] = {}
            current = current[letter]
        current[self.endSymbol] = word

    def contains(self, word):
        current = self.root
        for letter in word:
            if letter not in current:
                return False
            current = current[letter]
        return self.endSymbol in current

    def remove(self, word):
        current = self.root
        for letter in word:
            if letter not in current:
                return False
            current = current[letter]
        if self.endSymbol not in current:
            return False
        del current[self.endSymbol]
        return True
```


## Binary Search Tree

implement a binary search tree

```python
class BinarySearchTree:
    def __init__(self):
        self.root = None
        self.size = 0

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.root.__iter__()

    def put(self, key, val):
        if self.root:
            self._put(key, val, self.root)
        else:
            self.root = TreeNode(key, val)
        self.size = self.size + 1

    def _put(self, key, val, currentNode):
        if key < currentNode.key:
            if currentNode.hasLeftChild():
                self._put(key, val, currentNode.leftChild)
            else:
                currentNode.leftChild = TreeNode(key, val, parent=currentNode)
        else:
            if currentNode.hasRightChild():
                self._put(key, val, currentNode.rightChild)
            else:
                currentNode.rightChild = TreeNode(key, val, parent=currentNode)

    def __setitem__(self, k, v):
        self.put(k, v)

    def get(self, key):
        if self.root:
            res = self._get(key, self.root)
            if res:
                return res.payload
            else:
                return None
        else:
            return None

    def _get(self, key, currentNode):
        if not currentNode:
            return None
        elif currentNode.key == key:
            return currentNode
        elif key < currentNode.key:
            return self._get(key, currentNode.leftChild)
        else:
            return self._get(key, currentNode.rightChild)

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        if self._get(key, self.root):
            return True
        else:
            return False

    def delete(self, key):
        if self.size > 1:
            nodeToRemove = self._get(key, self.root)
            if nodeToRemove:
                self.remove(nodeToRemove)
                self.size = self.size-1
            else:
                raise KeyError('Error, key not in tree')
        elif self.size == 1 and self.root.key == key:
            self.root = None
            self.size = self.size - 1
        else:
            raise KeyError('Error, key not in tree')

    def __delitem__(self, key):
        self.delete(key)

    def spliceOut(self):
        if self.isLeaf():
            if self.isLeftChild():
                self.parent.leftChild = None
            else:
                self.parent.rightChild = None
        elif self.hasAnyChildren():
            if self.hasLeftChild():
                if self.isLeftChild():
                    self.parent.leftChild = self.leftChild
                else:
                    self.parent.rightChild = self.leftChild
                self.leftChild.parent = self.parent
            else:
                if self.isLeftChild():
                    self.parent.leftChild = self.rightChild
                else:
                    self.parent.rightChild = self.rightChild
                self.rightChild.parent = self.parent

    def findSuccessor(self):
        succ = None
        if self.hasRightChild():
            succ = self.rightChild.findMin()
        else:
            if self.parent:
                if self.isLeftChild():
                    succ = self.parent
                else:
                    self.parent.rightChild = None
                    succ = self.parent.findSuccessor()
                    self.parent.rightChild = self
        return succ

    def findMin(self):
        current = self
        while current.hasLeftChild():
            current = current.leftChild
        return current

    def remove(self, currentNode):
        if currentNode.isLeaf(): #leaf
            if currentNode == currentNode.parent.leftChild:
                currentNode.parent.leftChild = None
            else:
                currentNode.parent.rightChild = None
        elif currentNode.hasBothChildren(): #interior
            succ = currentNode.findSuccessor()
            succ.spliceOut()
            currentNode.key = succ.key
            currentNode.payload = succ.payload

        else: # this node has one child
            if currentNode.hasLeftChild():
                if currentNode.isLeftChild():
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.leftChild
                elif currentNode.isRightChild():
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.leftChild
                else:
                    currentNode.replaceNodeData(currentNode.leftChild.key,
                                    currentNode.leftChild.payload,
                                    currentNode.leftChild.leftChild,
                                    currentNode.leftChild.rightChild)
            else:
                if currentNode.isLeftChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.rightChild
                elif currentNode.isRightChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.rightChild
                else:
                    currentNode.replaceNodeData(currentNode.rightChild.key,
                                    currentNode.rightChild.payload,
                                    currentNode.rightChild.leftChild,
                                    currentNode.rightChild.rightChild)
```

## Red-Black Tree

implement a red-black tree

```python
class RedBlackTree:
    def __init__(self):
        self.root = None
        self.size = 0

    def length(self):
        return self.size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.root.__iter__()

    def put(self, key, val):
        if self.root:
            self._put(key, val, self.root)
        else:
            self.root = TreeNode(key, val)
            self.root.color = 'black'
        self.size = self.size + 1

    def _put(self, key, val, currentNode):
        if key < currentNode.key:
            if currentNode.hasLeftChild():
                self._put(key, val, currentNode.leftChild)
            else:
                currentNode.leftChild = TreeNode(key, val, parent=currentNode)
                self.fixTree(currentNode.leftChild)
        else:
            if currentNode.hasRightChild():
                self._put(key, val, currentNode.rightChild)
            else:
                currentNode.rightChild = TreeNode(key, val, parent=currentNode)
                self.fixTree(currentNode.rightChild)

    def __setitem__(self, k, v):
        self.put(k, v)

    def get(self, key):
        if self.root:
            res = self._get(key, self.root)
            if res:
                return res.payload
            else:
                return None
        else:
            return None

    def _get(self, key, currentNode):
        if not currentNode:
            return None
        elif currentNode.key == key:
            return currentNode
        elif key < currentNode.key:
            return self._get(key, currentNode.leftChild)
        else:
            return self._get(key, currentNode.rightChild)

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        if self._get(key, self.root):
            return True
        else:
            return False

    def delete(self, key):
        if self.size > 1:
            nodeToRemove = self._get(key, self.root)
            if nodeToRemove:
                self.remove(nodeToRemove)
                self.size = self.size-1
            else:
                raise KeyError('Error, key not in tree')
        elif self.size == 1 and self.root.key == key:
            self.root = None
            self.size = self.size - 1
        else:
            raise KeyError('Error, key not in tree')

    def __delitem__(self, key):
        self.delete(key)

    def spliceOut(self):
        if self.isLeaf():
            if self.isLeftChild():
                self.parent.leftChild = None
            else:
                self.parent.rightChild = None
        elif self.hasAnyChildren():
            if self.hasLeftChild():
                if self.isLeftChild():
                    self.parent.leftChild = self.leftChild
                else:
                    self.parent.rightChild = self.leftChild
                self.leftChild.parent = self.parent
            else:
                if self.isLeftChild():
                    self.parent.leftChild = self.rightChild
                else:
                    self.parent.rightChild = self.rightChild
                self.rightChild.parent = self.parent

    def findSuccessor(self):
        succ = None
        if self.hasRightChild():
            succ = self.rightChild.findMin()
        else:
            if self.parent:
                if self.isLeftChild():
                    succ = self.parent
                else:
                    self.parent.rightChild = None
                    succ = self.parent.findSuccessor()
                    self.parent.rightChild = self
        return succ

    def findMin(self):
        current = self
        while current.hasLeftChild():
            current = current.leftChild
        return current

    def remove(self, currentNode):
        if currentNode.isLeaf(): #leaf
            if currentNode == currentNode.parent.leftChild:
                currentNode.parent.leftChild = None
            else:
                currentNode.parent.rightChild = None
        elif currentNode.hasBothChildren(): #interior
            succ = currentNode.findSuccessor()
            succ.spliceOut()
            currentNode.key = succ.key
            currentNode.payload = succ.payload

        else: # this node has one child
            if currentNode.hasLeftChild():
                if currentNode.isLeftChild():
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.leftChild
                elif currentNode.isRightChild():
                    currentNode.leftChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.leftChild
                else:
                    currentNode.replaceNodeData(currentNode.leftChild.key,
                                    currentNode.leftChild.payload,
                                    currentNode.leftChild.leftChild,
                                    currentNode.leftChild.rightChild)
            else:
                if currentNode.isLeftChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.leftChild = currentNode.rightChild
                elif currentNode.isRightChild():
                    currentNode.rightChild.parent = currentNode.parent
                    currentNode.parent.rightChild = currentNode.rightChild
                else:
                    currentNode.replaceNodeData(currentNode.rightChild.key,
                                    currentNode.rightChild.payload,
                                    currentNode.rightChild.leftChild,
                                    currentNode.rightChild.rightChild)

    def fixTree(self, currentNode):
        if currentNode.parent == None:
            return
        if currentNode.parent.color == 'red' and currentNode.color == 'red':
            if currentNode.uncle() and currentNode.uncle().color == 'red':
                currentNode.parent.color = 'black'
                currentNode.uncle().color = 'black'
                currentNode.grandparent().color = 'red'
                self.fixTree(currentNode.grandparent())
            elif currentNode.uncle() == None or currentNode.uncle().color == 'black':
                if currentNode.isRightChild() and currentNode.parent.isLeftChild():
                    self.rotateLeft(currentNode.parent)
                    currentNode = currentNode.leftChild
                elif currentNode.isLeftChild() and currentNode.parent.isRightChild():
                    self.rotateRight(currentNode.parent)
                    currentNode = currentNode.rightChild
                if currentNode.isLeftChild():
                    self.rotateRight(currentNode.grandparent())
                else:
                    self.rotateLeft(currentNode.grandparent())
                currentNode.parent.color = 'black'
                currentNode.grandparent().color = 'red'
        self.root.color = 'black'

    def rotateLeft(self, rotRoot):
        newRoot = rotRoot.rightChild
        rotRoot.rightChild = newRoot.leftChild
        if newRoot.leftChild != None:
            newRoot.leftChild.parent = rotRoot
        newRoot.parent = rotRoot.parent
        if rotRoot.isRoot():
            self.root = newRoot
        else:
            if rotRoot.isLeftChild():
                rotRoot.parent.leftChild = newRoot
            else:
                rotRoot.parent.rightChild = newRoot
        newRoot.leftChild = rotRoot
        rotRoot.parent = newRoot

    def rotateRight(self, rotRoot):
        newRoot = rotRoot.leftChild
        rotRoot.leftChild = newRoot.rightChild
        if newRoot.rightChild != None:
            newRoot.rightChild.parent = rotRoot
        newRoot.parent = rotRoot.parent
        if rotRoot.isRoot():
            self.root = newRoot
        else:
            if rotRoot.isRightChild():
                rotRoot.parent.rightChild = newRoot
            else:
                rotRoot.parent.leftChild = newRoot
        newRoot.rightChild = rotRoot
        rotRoot.parent = newRoot
```
