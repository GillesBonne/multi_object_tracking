"""Class for handling the embedding database."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils import calc_cosine_sim, calc_distance


class EmbeddingsDatabase():
    """Class for handling the embedding database. Database consists of list of tuples
    that have the following structure: (id, calls_since_last_update, embedding_vector)."""

    def __init__(self, memory_length=15, memory_update=1, metric='Euclidean'):
        self.database = []  # Create empty database
        self.curr_max_id = 0  # Current highest identification number in the database

        self.memory_length = memory_length  # Length in frames to memorize the embeddings
        self.memory_update = memory_update  # Memory update value (0 is no update, 1 is replace)

        if metric == 'Euclidean':
            self.function = calc_distance
        elif metric == 'cosine':
            self.function = calc_cosine_sim
        else:
            raise Exception('Unknown metric function!')

        self.total_cost = 0
        self.num_samples = 0

    def update_database(self):
        """Update database by removing expired elements."""
        self.database = [(e[0], e[1]+1, e[2]) for e in self.database if e[1] < self.memory_length]

    def update_embedding(self, new_embedding, index):
        """Update single embedding in the database."""
        t = self.database[index]
        self.database[index] = (t[0],
                                0,
                                (1-self.memory_update) * t[2] + self.memory_update * new_embedding)
        return t[0]

    def add_embedding(self, new_embedding):
        """Add new embedding to the database."""
        new_embedding_id = self.curr_max_id
        self.curr_max_id += 1

        self.database.append((new_embedding_id, 0, new_embedding))
        return new_embedding_id

    def match_embeddings(self, new_embeddings, max_distance=0.1):
        """Match the embeddings in 'new_embeddings' with embeddings in the database."""
        self.update_database()  # Update the database and remove expired elements

        ids_list = []
        if not self.database:
            for new_embedding in new_embeddings:
                ids_list.append(self.add_embedding(new_embedding))
            return ids_list

        # Create cost matrix
        cost_matrix = np.empty([len(new_embeddings), len(self.database)])
        for i, new_embedding in enumerate(new_embeddings):
            for j, element in enumerate(self.database):
                cost_matrix[i, j] = self.function(new_embedding, element[2])

        #print(cost_matrix)

        # Use the Hugarian algorithm for unique assignment of ids
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        for row_index, new_embedding in enumerate(new_embeddings):
            if row_index in row_indices:
                col_index = col_indices[row_indices.tolist().index(row_index)]

                #print(cost_matrix[row_index, col_index])

                self.update_average_cost(cost_matrix[row_index, col_index])

                if cost_matrix[row_index, col_index] <= max_distance:
                    # Embedding is assigned and distance is not too large
                    ids_list.append(self.update_embedding(new_embedding, col_index))
                else:
                    # Embedding is assigned but distance is too large
                    ids_list.append(self.add_embedding(new_embedding))
            else:
                # Embedding is not assigned
                ids_list.append(self.add_embedding(new_embedding))

        return ids_list

    def update_average_cost(self, cost_value):
        """Update the total cost and number of samples."""
        self.total_cost += cost_value
        self.num_samples += 1

    def get_average_cost(self):
        """Return the average cost since last call."""
        avg_cost = self.total_cost / self.num_samples

        self.total_cost = 0  # Reset the total cost
        self.num_samples = 0  # Reset the number of samples

        return avg_cost
