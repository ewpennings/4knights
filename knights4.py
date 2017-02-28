from enum import Enum
from copy import deepcopy
from math import inf
from queue import PriorityQueue
from itertools import permutations


# List of possible moves for a knight
possible_moves = [p for p in permutations([-2, -1, 1, 2], 2) if abs(p[0] - p[1]) % 2]


# Representation of a square of the board.
# A square may be empty, contain a black knight or contain a white knight
class Square(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2


# Representation of a node
class Node:
    def __init__(self):
        self.board = list()
        self.came_from = None
        # Initialize with a default g_score and f_score of infinity
        self.g_score = inf
        self.f_score = inf

    def __eq__(self, other):
        return self.board == other.board

    def __lt__(self, other):
        return self.f_score < other.f_score

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        rpr = ""
        for row in self.board:
            for square in row:
                if square is None:
                    rpr += "   "
                elif square is Square.EMPTY:
                    rpr += "[ ]"
                elif square is Square.BLACK:
                    rpr += "[B]"
                elif square is Square.WHITE:
                    rpr += "[W]"
            rpr += "\n"
        return rpr

    # Read a board from a file
    def read(self, filename: str):
        result = False
        try:
            fh = open(filename)
            data = fh.read()
            row = []

            for c in data:
                if c == ' ':
                    row.append(None)
                elif c == '0':
                    row.append(Square.EMPTY)
                elif c == 'B':
                    row.append(Square.BLACK)
                elif c == 'W':
                    row.append(Square.WHITE)
                elif c == '\n':
                    self.board.append(row)
                    row = []
            if len(row):
              self.board.append(row)
            result = True
        except FileNotFoundError as e:
            print(e)
        return result

    # Return a Square at a specific position, or None if there is no Square at that position
    def square_at(self, x: int, y: int):
        if x < 0 or y < 0:
            return None
        try:
            return self.board[y][x]
        except IndexError:
            return None

    # Return the distance between this Node and another Node
    def distance_to(self, other):
        return 0 if self == other else 1

    # Create a goal Node ( a board where all pieces have switched color )
    def goal(self):
        goal = Node()
        goal.board = deepcopy(self.board)
        height = len(goal.board)
        for y in range(0, height):
            width = len(goal.board[y])
            for x in range(0, width):
                s = self.board[y][x]
                if s is Square.BLACK:
                    goal.board[y][x] = Square.WHITE
                elif s is Square.WHITE:
                    goal.board[y][x] = Square.BLACK
        return goal

    # Return a list of neighbours for this Node
    def neighbours(self):
        neighbours = []
        for y in range(0, len(self.board)):
            for x in range(0, len(self.board[y])):
                t = self.board[y][x]
                if t is not None and t is not Square.EMPTY:
                    for p in possible_moves:
                        pt = self.square_at(x + p[0], y + p[1])
                        if pt is Square.EMPTY:
                            neighbour = Node()
                            neighbour.board = deepcopy(self.board)
                            neighbour.board[y + p[1]][x + p[0]] = deepcopy(t)
                            neighbour.board[y][x] = deepcopy(pt)
                            if neighbour not in neighbours:
                                neighbours.append(neighbour)
        return neighbours

    # Return the estimated cost to reach another Node
    def heuristic_cost(self, other):
        cost = 0
        height = len(self.board)
        for y in range(0, height):
            width = len(self.board[y])
            for x in range(0, width):
                goal_square = other.board[y][x]
                if goal_square is not Square.EMPTY:
                    current_square = self.board[y][x]
                    if current_square is Square.EMPTY:
                        cost += 1
                    elif goal_square != current_square:
                        cost += 2
        return cost


# Return the path from
def reconstruct_path(node: Node):
    path = [node]
    while node.came_from:
        node = node.came_from
        path.append(node)
    path.pop()
    path.reverse()
    return path


# Implementation of the A* algorithm
def a_star(start: Node, goal: Node):
    closed_set = set()
    open_set = PriorityQueue()
    open_set.put(start)

    start.g_score = 0
    start.f_score = start.heuristic_cost(goal)

    while not open_set.empty():
        current = open_set.get()

        if current == goal:
            return reconstruct_path(current)

        closed_set.add(current)

        for neighbour in current.neighbours():
            if neighbour in closed_set:
                continue

            g_score = current.g_score + current.distance_to(neighbour)

            if neighbour not in open_set.queue:
                open_set.put(neighbour)
            elif g_score >= neighbour.g_score:
                continue

            neighbour.came_from = current
            neighbour.g_score = g_score
            neighbour.f_score = neighbour.g_score + neighbour.heuristic_cost(goal)

    # No path was found
    return None


def main():
    node = Node()
    if node.read("challenges/challenge4.txt"):
        print("Looking for solution...\nStart:\n{0}".format(node))
        path = a_star(node, node.goal())

        if path is None:
            print("No solution found")
        else:
            print("Solution found with {0} moves:".format(len(path)))
            for i in range(0, len(path)):
                print("Move {0}:\n{1}".format(i + 1, path[i]))


if __name__ == "__main__":
    main()
