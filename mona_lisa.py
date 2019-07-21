import cv2
import numpy as np
import random

num_inds = 20
num_genes = 50
tm_size = 5
frac_elites = 0.2
frac_parents = 0.6
mutation_prob = 0.2
mutation_type = "guided"


class Circle:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.radius = random.randint(0, height)
        self.center = (random.randint(0 - self.radius, self.radius + self.width),
                       random.randint(0 - self.radius, self.radius + self.height))
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.alpha = random.random()

    def __repr__(self):
        return "radius: {0}, color: {1}, center: {2}, alpha: {3}".format(self.radius, self.color, self.center, self.alpha)


class Individual:
    def __init__(self, index=None, fitness=None, size=num_genes):
        self.fitness = fitness
        self.index = index
        self.size = size
        self.list = []

    def __repr__(self):
        return "Individual: {0}, Fitness:{1}".format(self.index, self.fitness)

    def add_gene(self, item):
        self.list.append(item)

    def sort_genes(self):
        self.list = sorted(self.list, key=lambda k: k.radius, reverse=True)


class Population:
    def __init__(self, index=None, size=num_inds):
        self.size = size
        self.index = index
        self.list = []

    def __repr__(self):
        return "Population {0} with {1} individuals".format(self.index, self.size)

    def add_individual(self, item):
        self.list.append(item)

    def sort_population(self):
        self.list = sorted(self.list, key=lambda x: x.fitness)

    def delete_individual(self, item):
        self.list.remove(item)


def initialization(height, width, index):
    population = Population(index)
    for i in range(num_inds):
        individual = Individual(i)
        for j in range(num_genes):
            my_circle = Circle(height, width)
            individual.add_gene(my_circle)
        individual.sort_genes()
        population.add_individual(individual)
    return population


def evaluation(population):
    mona_lisa = cv2.imread('mona_lisa.jpg')
    # cv2.imshow('mona-lisa', mona_lisa)
    height, width, channels = mona_lisa.shape
    rgb_color = (255, 255, 255)
    my_monalisa = np.zeros((height, width, channels), np.uint8)
    my_monalisa[:] = rgb_color
    # cv2.imshow('my mona lisa', my_monalisa)
    for individual in population.list:
        for gene in individual.list:
            overlay = my_monalisa.copy()
            cv2.circle(overlay, center=gene.center,
                       radius=gene.radius, color=gene.color,
                       thickness=-1)
            my_monalisa = cv2.addWeighted(src1=overlay, alpha=gene.alpha, src2=my_monalisa,
                                          beta=1 - gene.alpha,
                                          gamma=0)
        fit = 0
        for k in range(channels):
            for j in range(width):
                for i in range(height):
                    # print(mona_lisa[i][j][k], my_monalisa[i][j][k])
                    fit += - (abs(int(mona_lisa[i][j][k]) - int(my_monalisa[i][j][k]))) ** 2

        individual.fitness = fit
    return population


def selection(population):
    elites = population.list[- int(frac_elites * num_inds):]
    next_generation = Population()
    for item in elites:
        population.delete_individual(item)
        next_generation.add_individual(item)

    idx = random.choices(population.list, k=tm_size)
    idx = sorted(idx, key=lambda x: x.fitness, reverse=True)
    # tournament_candidates = [tournament_applicants[item] for item in idx]
    p = 0.5
    probabilities = [p * ((1 - p) ** i) for i in range(tm_size)]
    winner = random.choices(idx, weights=probabilities, k=1)
    winner = winner[0]
    next_generation.add_individual(winner)
    return population, next_generation


def crossover(population)

def main():
    mona_lisa = cv2.imread('mona_lisa.jpg')
    # cv2.imshow('mona-lisa', mona_lisa)
    height, width, channels = mona_lisa.shape
    for i in range(10000):
        population = initialization(height, width, i)
        population = evaluation(population)
        population.sort_population()
        population, next_generation = selection(population)
    '''while True:
        Crossover()
        Mutation()'''

if __name__ == "__main__":
    main()
