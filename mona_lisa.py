import cv2
import numpy as np
import random
import webbrowser
import os.path
import sys


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
    def __init__(self, index=None, fitness=None, size=0, image=None):
        self.fitness = fitness
        self.index = index
        self.size = size
        self.list = []
        self.image = image

    def __repr__(self):
        return "Individual: {0}, Fitness:{1}".format(self.index, self.fitness)

    def add_gene(self, item):
        self.list.append(item)
        self.size += 1

    def sort_genes(self):
        self.list = sorted(self.list, key=lambda k: k.radius, reverse=True)

    def swap_genes(self, other, index):
        self.list[index], other.list[index] = other.list[index], self.list[index]


class Population:
    def __init__(self, index=None, size=0):
        self.size = size
        self.index = index
        self.list = []

    def __repr__(self):
        return "Population {0} with {1} individuals".format(self.index, self.size)

    def add_individual(self, item):
        self.list.append(item)
        self.size += 1

    def sort_population(self):
        self.list = sorted(self.list, key=lambda x: x.fitness)

    def delete_individual(self, item):
        self.list.remove(item)
        self.size -= 1

    def update_index(self, index):
        self.index = index


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
        individual.image = my_monalisa
        '''cv2.imshow('my mona lisa', my_monalisa)
        cv2.waitKey(0)'''
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
    elite_individuals = Population()
    for item in elites:
        population.delete_individual(item)
        elite_individuals.add_individual(item)

    idx = random.choices(population.list, k=tm_size)
    idx = sorted(idx, key=lambda x: x.fitness, reverse=True)
    # tournament_candidates = [tournament_applicants[item] for item in idx]
    p = 0.5
    probabilities = [p * ((1 - p) ** i) for i in range(tm_size)]
    winner = random.choices(idx, weights=probabilities, k=1)
    winner = winner[0]
    population.delete_individual(winner)
    return population, winner, elite_individuals


def crossover(population):
    for j in range(population.size - 1, population.size - int(population.size * frac_parents) - 1, -1):
        if j is not population.size - int(population.size * frac_parents):
            for k in range(num_genes):
                prob = random.random()
                if prob < 0.5:
                    population.list[j].swap_genes(other=population.list[j - 1], index=k)
        population.list[j].fitness = None
    return population


def mutation(population):

    for individual in population.list:
        probability = random.random()
        if probability < mutation_prob:
            selected_gene = random.choice(individual.list)
            if mutation_type is "guided":
                selected_gene.center = (random.randint(selected_gene.center[0] - selected_gene.width // 4,
                                                       selected_gene.center[0] + selected_gene.width // 4),
                                        random.randint(selected_gene.center[1] - selected_gene.height // 4,
                                                       selected_gene.center[1] + selected_gene.height // 4))
                selected_gene.radius = random.randint(max(0, selected_gene.radius - 10), selected_gene.radius + 10)
                selected_gene.color = (random.randint(max(0, selected_gene.color[0] - 64),
                                                      min(255, selected_gene.color[0] + 64)),
                                       random.randint(max(0, selected_gene.color[1] - 64),
                                                      min(255, selected_gene.color[1] + 64)),
                                       random.randint(max(0, selected_gene.color[2] - 64),
                                                      min(255, selected_gene.color[2] + 64)))
                selected_gene.alpha = random.uniform(max(0, selected_gene.alpha - 0.25), min(1, selected_gene.alpha + 0.25))
            if mutation_type is "unguided":
                selected_gene.center = Circle(selected_gene.height, selected_gene.width)

    return population


def main():
    path = os.getcwd()
    # open a web page with the image that refresh every second
    if os.path.exists(path + "/file.html") and os.path.exists(path + "/script.js"):
        webbrowser.open(path + "/file.html", "chrome")
    else:
        print("\nPut the \".html\" and the \".js\" here: " + os.path.dirname(path))
        sys.exit()
    mona_lisa = cv2.imread('mona_lisa.jpg')
    # cv2.imshow('mona-lisa', mona_lisa)
    height, width, channels = mona_lisa.shape
    population = initialization(height, width, 0)
    for i in range(10000):
        population.update_index(i)
        for individual in population.list:
            individual.sort_genes()
        population = evaluation(population)
        population.sort_population()
        best = population.list[-1].image
        cv2.imwrite("pic.png", best)
        fitness = [individual.fitness for individual in population.list]
        print("Generations: " + str(i))
        print(fitness)
        population, tournament_winner, elite_individuals = selection(population)
        crossover(population)
        population.add_individual(tournament_winner)
        population = mutation(population)
        for elite in elite_individuals.list:
            population.add_individual(elite)


if __name__ == "__main__":
    main()
