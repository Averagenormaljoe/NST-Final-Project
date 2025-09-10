class BestImage:
    def __init__(self, image, cost : float, iterations : int):
        self.image = image
        self.cost = cost
        self.iterations = iterations
    def __str__(self):
        return f"BestImage(cost={self.cost}, iterations={self.iterations})"
    def get_image(self):
        return self.image
    def get_cost(self) -> float:
        return self.cost
    def get_iterations(self) -> int:
        return self.iterations