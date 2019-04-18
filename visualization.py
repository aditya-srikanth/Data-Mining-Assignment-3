import matplotlib.pyplot as plt
import math

class Visualization:
    '''
    @Author: Naren Surampudi
    '''

    def __init__(self):
        self.OUTLIERS = []
        self.NON_OUTLIERS = []

    def dimension_reduction(self, point):
        temp_point = []
        reduced_point = [0,0]
        index = 1
        for element in point:
            if not math.isnan(element % index):
                temp_point.append(element % index)
            index = index + 1

        for element in temp_point:
            if element % 2 == 0:
                reduced_point[1] = reduced_point[1] + element
            else:
                reduced_point[0] = reduced_point[0] + element

        reduced_point[0] = round(reduced_point[0], 2)
        reduced_point[1] = round(reduced_point[1], 2)

        return reduced_point

    def outlier_plot(self):
        for element in self.OUTLIERS:
            plt.scatter(element[0], element[0], facecolors='none', edgecolors='r', marker='o')
        for element in self.NON_OUTLIERS:
            plt.scatter(element[0], element[1], facecolors='none', edgecolors='b', marker = 'o')

        plt.show()
