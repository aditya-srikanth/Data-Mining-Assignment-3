import matplotlib.pyplot as plt
import math

class Visualization:
    """
    This class contains methods for reducing the dimensions of the points to 2-D and
    visualization of the reduced points.
    Attributes
    ----------
    OUTLIERS : list
        List of points marked as outliers.
    NON_OUTLIERS : list
        List of points that are not marked as outliers.
    """

    def __init__(self):
        self.OUTLIERS = []
        self.NON_OUTLIERS = []
        self.K = 1

    def dimension_reduction(self, point):
        """
        This method is used for reducing the dimensions of the given point to 2-D.
        Parameters
        ----------
        point : list
            A list of coordinates representing an n-dimensional vector.
        Returns
        -------
        type list
            A list representing a 2-D point in the x-y plane.
        """
        temp_point = []
        reduced_point = [0,0]
        index = 1
        for element in point:
            if not math.isnan(element % index):
                #   Using modulo operation to spread values of coordinates.
                temp_point.append(element % index)
            index = index + 1

        for element in temp_point:
            #   The modulo results are distributed among the two coordinates according to
            #   their divisibilty by 2.
            if element % 2 == 0:
                reduced_point[1] = reduced_point[1] + element
            else:
                reduced_point[0] = reduced_point[0] + element

        reduced_point[0] = round(reduced_point[0], 2)
        reduced_point[1] = round(reduced_point[1], 2)

        return reduced_point

    def outlier_plot(self):
        """
        This mehtod takes the points marked as outliers and non-outliers and plots them as
        a scatter plot.
        Returns
        -------
        None
            The result of this method is a matplotlib scatter plot.
        """
        for element in self.OUTLIERS:
            plt.scatter(element[0], element[0], facecolors='none', edgecolors='r', marker='o')
        for element in self.NON_OUTLIERS:
            plt.scatter(element[0], element[1], facecolors='none', edgecolors='b', marker = 'o')

        plt.xlabel("K = " + str(self.K))
        plt.show()