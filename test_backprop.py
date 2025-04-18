from typing import List, Optional, Set
from collections import deque
import pytest

class Val:
    def __init__(self, data, children=()):
        self.data = data
        self.children = set(children)
        self._backward = lambda: None
        self.grad = 0.0

    def backward_with_zero_grad(self):
        nodes: List["Val"]= []
        visited = set()
        queue = deque([self])
        while queue:
            n = len(queue)
            for _ in range(n):
                curr = queue.pop()
                nodes.append(curr)
                visited.add(curr)
                for child in curr.children:
                    if child not in visited:
                        queue.appendleft(child)
        self.grad = 1.0
        for node in nodes:
            node._backward()
        for node in nodes:
            node.grad = 0.0
            
    def __mul__(self, other):
        othernode = other if isinstance(other, Val) else Val(data=other)
        out = Val(self.data * othernode.data, children=())
        def backward():
            self.grad += othernode.data * out.grad
            othernode.grad += self.data * out.grad
        out._backward = backward
        return out
        

def test():
    pass
    

if __name__ == "__main__":
    pytest.main()