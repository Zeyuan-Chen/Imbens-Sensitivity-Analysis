import numpy as np 
import pandas as pd


def gen_Exogenous(name, n, m_cont = 1, m_cata = 0, cont_params = [[0,1]], cata_params = []):
    #name:   column name prefix
    #n:      number of samples
    #m_cont: number of continuouse features
    #m_cata: number of catagorical features
    #cont_params: a list of length m_cont, each element is mean and std, the feature will be drawn i.i.d from N(mean, std) 
    #cata_params: a list of length m_cata, each element is a list of ps for each catagory, feature will be drawn i.i.d from multi(ps)

    assert len(cont_params) == m_cont,  "continuous var different length"
    assert len(cata_params) == m_cata,  "catagorical var different length" 
    Xs = []
    for j in range(len(cont_params)):
        cont_param = cont_params[j]
        X = np.random.normal(cont_param[0],cont_param[1], n)
        Xs.append(np.random.normal(cont_param[0],cont_param[1], n))

    for j in range(len(cata_params)):
        cata_param = cata_params[j]
        X = np.random.multinomial(1, cata_param, n)
        X = np.argmax(X, axis = 1)
        Xs.append( X)

    Xs = pd.DataFrame(np.array(Xs).T, 
                      index = [f"sample_{i}" for i in range(n)],
                      columns = [f"{name}_cont_{j}" for j in range(len(cont_params))] + [f"{name}_cata_{j}" for j in range(len(cata_params))])
    return Xs


def get_ATE(full_df, condition_cols, Treatment_col, Outcome_col):
    stratas = full_df.groupby(condition_cols).mean().index

    strata_ps = full_df.groupby(condition_cols).count().values[:,0]
    strata_ps = strata_ps/sum(strata_ps)

    strata_ATEs = []
    for strata in stratas:
    #strata = stratas[0]    
        strata_df = full_df[np.all((full_df[condition_cols] == strata).values, axis = 1)]
        strata_ATE = np.mean(strata_df[strata_df[Treatment_col] == 1][Outcome_col]) - np.mean(strata_df[strata_df[Treatment_col] == 0][Outcome_col])
        strata_ATEs = strata_ATEs + [strata_ATE]

    ATE = np.sum(np.array(strata_ATEs) * strata_ps)
    return ATE