# http://www.stokastik.in/using-kd-tree-for-nearest-neighbor-search/
    
import numpy as np
import time, math, heapq
from collections import deque

class QueueObj(object):
    """Class for defining all variables that goes into the queue while constructing the KD Tree"""
    def __init__(self, indices, depth, node, left, right):
        self.indices = indices
        self.depth = depth
        self.node = node
        self.left = left
        self.right = right

class Node(object):
    """Class for defining the node properties for the KD Tree"""
    def __init__(self, vector, split_value, split_row_index):
        self.vector = vector
        self.split_value = split_value
        self.split_row_index = split_row_index
        self.left = None
        self.right = None

class KDTree(object):
    """KD Tree class starts here"""
    def __init__(self, vectors):
        self.vectors = vectors
        self.root = None
        self.vector_dim = vectors.shape[1]
        
    def search(self, vector):
        node = self.root
        depth = 0
        while node is not None:
            if np.array_equal(node.vector, vector):
                return True
            axis = depth % self.vector_dim
            if vector[axis] <= node.split_value:
                node = node.left
            else:
                node = node.right
            depth += 1
        return False
    
    def construct(self):
        n = self.vectors.shape[0]
        queue = deque([QueueObj(range(n), 0, None, 0, 0)])
        while len(queue) > 0:
            qob = queue.popleft()
            q_front, depth, parent, l, r = qob.indices, qob.depth, qob.node, qob.left, qob.right
            axis = depth % self.vector_dim
            vectors = np.argsort(self.vectors[q_front, :][:, axis])
            vectors = [q_front[vec] for vec in vectors]
            m = len(vectors)
            median_index = int(m / 2)
            split_value = self.vectors[vectors[median_index]][axis]
            left, right = median_index + 1, m - 1
            while left <= right:
                mid = int((left + right) / 2)
                if self.vectors[vectors[mid]][axis] > split_value:
                    right = mid - 1
                else:
                    left = mid + 1
            median_index = left - 1
            node = Node(self.vectors[vectors[median_index]], split_value, vectors[median_index])
            if parent is None:
                self.root = node
            else:
                if l == 1:
                    parent.left = node
                else:
                    parent.right = node
            if median_index > 0:
                queueObj = QueueObj(vectors[:median_index], depth + 1, node, 1, 0)
                queue.append(queueObj)
            if median_index < m - 1:
                queueObj = QueueObj(vectors[median_index + 1:], depth + 1, node, 0, 1)
                queue.append(queueObj)
    
    def insert_distance_into_heap(self, distances, node, node_distance, k):
        if len(distances) == k and -distances[0][0] > node_distance:
            heapq.heappop(distances)
        if len(distances) < k:
            heapq.heappush(distances, (-node_distance, node.split_row_index))
            
    def nearest_neighbor(self, vector, k):
        search_stack = [(self.root, 0)]
        distances, visited = [], set()
        while len(search_stack) > 0:
            node, depth = search_stack[-1]
            axis = depth % self.vector_dim
            child_node = None
            if vector[axis] <= node.split_value:
                if node.left is None or node.left.split_row_index in visited:
                    node_distance = math.sqrt(np.sum((node.vector - vector) ** 2))
                    if node.right is None or node.right.split_row_index in visited:
                        self.insert_distance_into_heap(distances, node, node_distance, k)
                    else:
                        w = node_distance if len(distances) == 0 else - distances[0][0]
                        if node.split_value - vector[axis] <= w:
                            child_node = node.right
                else:
                    child_node = node.left
            else:
                if node.right is None or node.right.split_row_index in visited:
                    node_distance = math.sqrt(np.sum((node.vector - vector) ** 2))
                    if node.left is None or node.left.split_row_index in visited:
                        self.insert_distance_into_heap(distances, node, node_distance, k)
                    else:
                        w = node_distance if len(distances) == 0 else - distances[0][0]
                        if vector[axis] - node.split_value <= w:
                            child_node = node.left
                else:
                    child_node = node.right
            if child_node is None or child_node.split_row_index in visited:
                visited.add(node.split_row_index)
                search_stack.pop()
            else:
                search_stack.append((child_node, depth + 1))
        distances = [(-x, y) for x, y in distances]
        distances = sorted(distances, key=lambda k: k[0])
        return distances
