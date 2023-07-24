import random
import numpy as np
from collections import Counter
from EA.strategies import Combiner
from utilities import get_opr_name, get_inverse_weights

def test_weighted_random_selection(seen_oprs=[]):
    while True:
        single_ops_filtered = [opr_info for opr_info in Combiner.single_ops if not (
                get_opr_name(opr_info[0]) in seen_oprs and get_opr_name(opr_info[0]) in Combiner.one_time_oprs)]

        mutation_seen_oprs = [seen_opr for seen_opr in seen_oprs if seen_opr in Combiner.single_ops_name]
        if len(mutation_seen_oprs) > 0:
            counter_seen_oprs = dict(Counter(mutation_seen_oprs))
            weights_seen_oprs = get_inverse_weights(counter_seen_oprs, single_ops_filtered)
            weights = weights_seen_oprs.values()
            index = random.choices(np.arange(len(single_ops_filtered)), weights=weights)[0]
            opr = get_opr_name(single_ops_filtered[index][0])
            seen_oprs.append(opr)
        else:
            opr = random.choice(single_ops_filtered)[0]
            seen_oprs.append(opr)