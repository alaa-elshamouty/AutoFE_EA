from enum import IntEnum
from functools import partial
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, normalize, PolynomialFeatures

from utilities import dim_check


class Combiner:
    # TODO later adapt to autosklearn preprocessings
    single_ops = [StandardScaler(), QuantileTransformer(), PowerTransformer(), partial(normalize, axis=0), np.log,
                  np.delete, np.power]  # Todo remove power transform because TabPFN does it?
    combine_ops = [PolynomialFeatures(2), np.add, np.subtract, np.multiply, np.divide]

class Recombination(IntEnum):
    """Enum defining the recombination strategy choice"""

    NONE = -1  # can be used when only mutation is required
    UNIFORM = 0  # uniform crossover (only really makes sense for function dimension > 1)
    WEIGHTED = 1  # intermediate recombination

    @staticmethod
    def apply_recombination(opr,x,col_id,partner_x,partner_col_id,lower=-np.inf,upper= np.inf,max_dims=100):
        if "sklearn" in str(type(opr)):
            new_x = opr.fit_transform(x)
        else:
            col = x[:,col_id]
            partner_col = partner_x[:,partner_col_id]
            if opr.__name__ == 'divide': # to handle dividing by zero
                new_col = opr(col, partner_col, out=np.zeros_like(col), where=partner_col!=0).reshape(-1,1)
            else:
                new_col = opr(col,partner_col).reshape(-1,1)
            new_x = np.hstack((x,new_col))

        new_x = dim_check(new_x,lower,upper,max_dims)
        return new_x


class Mutation(IntEnum):
    """Enum defining the mutation strategy choice"""

    NONE = -1  # Can be used when only recombination is required
    UNIFORM = 0  # Uniform mutation
    WEIGHTED = 1  # Gaussian mutation

    @staticmethod
    def apply_mutation(opr,x,col_id):
        col = x[:,col_id].reshape(-1,1)
        if "sklearn" in str(type(opr)):
            x[:,col_id] = opr.fit_transform(col).squeeze()
        elif "partial" not in str(type(opr)):
            if opr.__name__ == 'delete':
                x = opr(x,col_id,axis=1)
            elif opr.__name__ == 'power':
                x[:,col_id] = opr(col,2).squeeze()
        else:
            x[:,col_id] = opr(col).squeeze()
        return x

class ParentSelection(IntEnum):
    """Enum defining the parent selection choice"""

    NEUTRAL = 0
    FITNESS = 1
    TOURNAMENT = 2


def apply_trajectory(dataset,trajectory):
    if trajectory == (None,None,None,None,None):
        return dataset
    traj_current_member = trajectory[0]
    traj_first_member = trajectory[1]
    traj_second_member= trajectory[2]
    if traj_first_member == (None,None,None,None,None) and traj_second_member == (None,None,None,None,None):
        new_x = apply_operator(traj_current_member,dataset,dataset,rec=True)
        return new_x
    elif traj_first_member == (None,None,None,None,None) and not traj_second_member:
        new_x = apply_operator(traj_current_member,dataset,None,rec=False)
        return new_x

    first_member_data = apply_trajectory(dataset,traj_first_member)
    if traj_second_member:
        second_member_data = apply_trajectory(dataset,traj_second_member)
        opr,_,col_id,_,partner_col_id = traj_current_member
        new_x = Recombination.apply_recombination(opr,first_member_data,col_id,second_member_data,partner_col_id)
    else:
        opr,_,col_id,_,_ = traj_current_member
        new_x =  Mutation.apply_mutation(opr,first_member_data,col_id)

    return new_x


def apply_operator(traj,data,partner_data,rec=False):
    new_x = data.copy()
    opr,_,col_id, _,partner_col_id = traj
    if rec:
        new_x = Recombination.apply_recombination(opr,new_x,col_id,partner_data,partner_col_id)
    else:
        new_x = Mutation.apply_mutation(opr,new_x,col_id)

    return new_x
