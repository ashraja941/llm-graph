class Graph:
    def __init__(self) -> None:
        self.nodes = {} 
        self.edges : set[tuple[str,str]] = set()
        self.branches = dict()

