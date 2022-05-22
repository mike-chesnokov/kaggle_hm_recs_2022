from typing import Tuple, Dict, List
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix


def get_user_item_matrix(user_item_log_: List[Tuple[str, int, int]], 
                         data_format: str = 'bin'):
    """
    Method for making user_item csr_matrix from user_item_log_.
    2 data formats: 
        - 'bin': user-item -> 1 or 0
        - 'cnt': user-item -> count of such pair
    
    Params:
        user_item_log_: list, (user, article, cnt_articles)
        data_format: str in ['bin', 'cnt']
    """
    # create user-row and item-col dicts
    users = []
    items = []
    data = []

    user_index = {}
    user_ind = 0

    item_index = {}
    item_ind = 0

    for pair in user_item_log_:
        if pair[0] not in user_index:
            user_index[pair[0]] = user_ind
            user_ind += 1

        if pair[1] not in item_index:
            item_index[pair[1]] = item_ind
            item_ind += 1

        users.append(user_index[pair[0]])
        items.append(item_index[pair[1]])
        if data_format == 'bin':
            data.append(1)
        elif data_format == 'cnt':
            data.append(pair[2])
            
    return user_index, item_index, csr_matrix((data, (users, items)))


def build_lookup_array(index_item: Dict[int, int]) -> np.ndarray:
    """
    Returns array of string IDs where index in array is numerical ID.

    :param index_item: mapping of numerical IDs to string IDs
    :return: array of string IDs, index in array is corresponding numerical ID
    """
    # pairs of [(numerical ID, string ID)], where numerical IDs are sorted
    pairs: List[Tuple[int, int]] = sorted(index_item.items(),
                                          key=lambda x: x[0],
                                          reverse=False)
    lookup_array = np.array([pair[1]
                             for pair in pairs])
    return lookup_array


def blend_product_lists(*cands, num_candidates=12):
    """Method for submissions blending"""
    cnt_dict = defaultdict(int)
    for cand in cands:
        # print(cands)
        for ind, pr in enumerate(cand):
            cnt_dict[pr] += ind + 1

    # add rank for items not any of set
    # for cand in cands:
    #    cand = set(cand)
    max_rank = max([len(el) for el in cands])

    for pr in cnt_dict:
        for cand in cands:
            if pr not in cand:
                cnt_dict[pr] += max_rank

    sorted_list = sorted(cnt_dict.items(), key=lambda x: x[1], reverse=False)

    return [el[0] for el in sorted_list[:num_candidates]]
