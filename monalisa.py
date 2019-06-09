# -*- coding: utf-8 -*-
"""
Created on Thu May  2 02:38:49 2019

@author: ferdi
"""

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

num_generations = 10000
num_inds = 20
num_genes = 250
tm_size = 5
frac_elites = 0.2
frac_parents = 0.6
mutation_prob = 0.2
mutation_type = 'guided'


''' Read the source image'''
img = cv2.imread("mona_lisa.jpg")
height, width, channels = img.shape


def main():

    ''' initialize population with num_inds individuals each having num_genes genes'''
    dtype = [('x', int), ('y', int), ('radius', int), ('r', int), ('g', int), ('b', int), ('alpha', float)]
    inds = np.zeros([num_inds, num_genes, 7])
    genes = np.zeros([1, 7])
    d = 0
    fitness_generation = np.empty([num_generations, num_inds])
    final_fitness = np.empty([num_generations])
    elites_index = np.empty([int(frac_elites*num_inds)])

    for i in range(0, num_inds):
        for j in range(0, num_genes):
            genes[:, 0] = random.randint(0, height)
            genes[:, 1] = random.randint(0, width)
            genes[:, 2] = random.randint(0, width)
            genes[:, 3] = random.randint(0, 255)
            genes[:, 4] = random.randint(0, 255)
            genes[:, 5] = random.randint(0, 255)
            genes[:, 6] = random.uniform(0, 1)
            inds[i, j] = genes

    ''' sort the genes '''
    for l in range(0,num_inds):
        a = sorted(inds[l], key=lambda a_entry: a_entry[2], reverse=True)
        a = np.asarray(a)
        inds[l] = a

 

    ''' while not all generations (num_generations) are computed:'''
    for k in range(0, num_generations):
        if np.mod(k, 1000) == 0:
            draw = 'true'
        elif k == 9999:
            draw = 'true'
        else:
            draw = 'false'
        print('generations:' + str(k))
        'Evaluate all the individuals'
        fitness_generation[k] = evaluation(inds, draw, d)
        print(fitness_generation[k])
        draw = 'false'
        d = d + 1
        fitness_generation_copy = fitness_generation[k]
        'Select individuals'
        elites_index, tourn_index = selection(fitness_generation[k])
        inds_copy = np.delete(inds, elites_index, axis=0)
        fitness_generation_copy = np.delete(fitness_generation_copy, elites_index, axis=0)
        'Do crossover on some individuals'
        childs, inds_copy = crossover(fitness_generation_copy, inds_copy)
        childs = np.asarray(childs)
        try:
            inds_copy = np.concatenate((childs, inds_copy), axis=0)
        except ValueError:
            inds_copy = childs
        'Mutate some individuals'
        inds_copy = mutation(inds_copy)
        inds = np.concatenate((inds_copy, inds[elites_index]), axis=0)
        fitness_index = np.where(fitness_generation[k] == np.amax(fitness_generation[k]))
        final_fitness[k] = fitness_generation[k, fitness_index[0][0]]


    plt.figure()
    plt.plot(final_fitness)
    plt.savefig('fitness_plot_1.png')

    plt.figure()
    plt.plot(final_fitness[99:999])
    plt.savefig('fitness_plot_2.png')


def evaluation(inds, draw, d):
    fitness_array = np.zeros([num_inds])
    for i in range(0, num_inds):
        rgb_color = (255, 255, 255)
        image = np.zeros((height, width, channels), np.uint8)
        image[:] = rgb_color
        overlay = image
        for j in range(0, num_genes):
            cv2.circle(overlay, center=(int(inds[i, j, 0]), int(inds[i, j, 1])), radius=int(inds[i, j, 2]),
                       color=(int(inds[i, j, 3]), int(inds[i, j, 4]), int(inds[i, j, 5])), thickness=-1)
            image = cv2.addWeighted(src1=overlay, alpha=inds[i, j, 6], src2=image, beta=1-inds[i, j, 6], gamma=0)
        fitness_array[i] = fitness(image)

    index_image = np.where(fitness_array == np.amax(fitness_array))
    index_image = index_image[0][0]
    if draw == 'true':
        rgb_color = (255, 255, 255)
        image = np.zeros((height, width, channels), np.uint8)
        image[:] = rgb_color
        image_2 = image
        for j in range(0, num_genes):
            #image_2 = image
            cv2.circle(image_2, center=(int(inds[index_image, j, 0]), int(inds[index_image, j, 1])), radius=int(inds[index_image, j, 2]),
                       color=(int(inds[index_image, j, 3]), int(inds[index_image, j, 4]), int(inds[index_image, j, 5])), thickness=-1)
            image = cv2.addWeighted(image_2, inds[index_image, j, 6], image, 1 - inds[index_image, j, 6], 0)

        a = str(d)
        cv2.imwrite(a + '.png', image)

    return fitness_array


def fitness(image):
    fit = 0
    fitness_ind = 0
    fitness_channel = 0
    for k in range(0, channels):
        fitness_ind = fitness_ind - fitness_channel
        for j in range(0, height):
            fitness_channel = fitness_channel + fit
            for i in range(0, width):
                pixel_src = int(img[j, i, k])
                pixel_image = int(image[j, i, k])
                fit = fit + (pixel_src - pixel_image)**2
    return fitness_ind


def selection(fitness_gen):
    fitness_index = np.empty([len(fitness_gen), 1])
    index_list = []
    tourn_list = []
    temp_array = np.empty(len(fitness_gen))
    for k in range(0, len(fitness_gen)):
        temp_array[k] = k
    for i in range(0, int(frac_elites * len(fitness_gen))):
        fitness_index = np.argsort(fitness_gen)
        index_list.append(fitness_index[-i-1])
        temp_array = np.delete(temp_array, index_list, axis=0)
    best = int(np.random.choice(temp_array, 1))
    for j in range(0, tm_size):
        ind = int(np.random.choice(temp_array, 1))
        if fitness_gen[ind] > fitness_gen[best]:
            best = ind
    tourn_list.append(best)

    return index_list, tourn_list


def crossover(fitness_generation_copy, inds_copy):

    fitness_index = np.argsort(fitness_generation_copy)
    parents_list_index = []
    childs = []
    sorrted = np.arange(0, frac_parents*num_inds)
    np.random.shuffle(sorrted)
    delete_index = []
    for i in range(0, int(frac_parents*num_inds)):
        parents_list_index.append(fitness_index[-i-1])

    for j in range(0, int(len(parents_list_index)/2)):
        p1_sel, p2_sel = sorrted[2*j], sorrted[2*j+1]
        p1_sel = int(p1_sel)
        p2_sel = int(p2_sel)
        parent1 = inds_copy[parents_list_index[p1_sel]]
        #print(parent1[0])
        parent2 = inds_copy[parents_list_index[p2_sel]]
        child1 = np.zeros([num_genes, 7])
        child2 = np.zeros([num_genes, 7])
        #print(parent2[0])

        delete_index.append(p1_sel)
        delete_index.append(p2_sel)
        for k in range(0, num_genes):
            crossover_point = random.randint(0, 1)
            if crossover_point == 1:
                child1[k] = parent2[k]
                child2[k] = parent1[k]
            else:
                child2[k] = parent2[k]
                child1[k] = parent1[k]

        childs.append(child1)
        childs.append(child2)

    inds_copy = np.delete(inds_copy, [delete_index], axis=0)

    return childs, inds_copy


def mutation(inds_copy):
    for k in range(0, num_inds):
        x = random.uniform(0, 1)
        if x < mutation_prob:
            i = random.randint(0, len(inds_copy)-1)
            j = random.randint(0, num_genes-1)
            if mutation_type == 'guided':
                inds_copy[i, j, 0] = random.randint(inds_copy[i, j, 0] - int(height / 4), inds_copy[i, j, 0] + int(height / 4))
                if inds_copy[i, j, 0] < 0:
                    inds_copy[i, j, 0] = 0
                elif inds_copy[i, j, 0] > height:
                    inds_copy[i, j, 0] = height
                inds_copy[i, j, 1] = random.randint(inds_copy[i, j, 1] - int(width / 4), inds_copy[i, j, 1] + int(width / 4))
                if inds_copy[i, j, 1] < 0:
                    inds_copy[i, j, 1] = 0
                elif inds_copy[i, j, 1] > width:
                    inds_copy[i, j, 1] = width
                inds_copy[i, j, 2] = random.randint(inds_copy[i, j, 2] - 10, inds_copy[i, j, 2] + 10)
                if inds_copy[i, j, 2] < 0:
                    inds_copy[i, j, 2] = 0
                elif inds_copy[i, j, 2] > width:
                    inds_copy[i, j, 2] = width
                inds_copy[i, j, 3] = random.randint(inds_copy[i, j, 3] - 64, inds_copy[i, j, 3] + 64)
                if inds_copy[i, j, 3] < 0:
                    inds_copy[i, j, 3] = 0
                elif inds_copy[i, j, 3] > 255:
                    inds_copy[i, j, 3] = 255
                inds_copy[i, j, 4] = random.randint(inds_copy[i, j, 4] - 64, inds_copy[i, j, 4] + 64)
                if inds_copy[i, j, 4] < 0:
                    inds_copy[i, j, 4] = 0
                elif inds_copy[i, j, 4] > 255:
                    inds_copy[i, j, 4] = 255
                inds_copy[i, j, 5] = random.randint(inds_copy[i, j, 5] - 64, inds_copy[i, j, 5] + 64)
                if inds_copy[i, j, 5] < 0:
                    inds_copy[i, j, 5] = 0
                elif inds_copy[i, j, 5] > 255:
                    inds_copy[i, j, 5] = 255
                inds_copy[i, j, 6] = random.uniform(inds_copy[i, j, 6] - 0.25, inds_copy[i, j, 6] + 0.25)
                if inds_copy[i, j, 6] < 0:
                    inds_copy[i, j, 6] = 0
                elif inds_copy[i, j, 6] > 1:
                    inds_copy[i, j, 6] = 1
            else:
                for i in range(0, len(inds_copy)):
                    for j in range(0, num_genes):
                        inds_copy[i, j, 0] = random.randint(0, height)
                        inds_copy[i, j, 1] = random.randint(0, width)
                        inds_copy[i, j, 2] = random.randint(0, width)
                        inds_copy[i, j, 3] = random.randint(0, 255)
                        inds_copy[i, j, 4] = random.randint(0, 255)
                        inds_copy[i, j, 5] = random.randint(0, 255)
                        inds_copy[i, j, 6] = random.uniform(0, 1)
    return inds_copy


if __name__ == "__main__":
    main()