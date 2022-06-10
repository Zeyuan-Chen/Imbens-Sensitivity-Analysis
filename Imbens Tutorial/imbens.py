import numpy as np 
import pandas as pd

#optimization
from tqdm import tqdm
from scipy.optimize import minimize
from scipy import interpolate

#visualization
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#core MLE
def imbens_obj(theta, *args):
    #theta should be [gamma, tau, beta, sigma]
    #print("please make sure U has only one feature and is bernoulli with p = 0.5")
    param_dict   = args[0]
    X            = param_dict['X'] #mat
    T            = param_dict['T'] #mat
    Y            = param_dict['Y'] #mat

    n   = X.shape[0]
    m_X = X.shape[1] 
    m_U = 1

    gamma        = np.array(theta[0:m_X])
    alpha        = param_dict['alpha']

    tau          = np.array(theta[m_X:(m_X + m_U)])
    beta         = np.array(theta[(m_X + m_U):(m_X + m_U + m_X)])
    delta        = param_dict['delta']
    sigma_square = theta[-1]**2


    coeff1 = 1/np.sqrt(2 * np.pi * sigma_square)
    coeff2 = 1/(2* sigma_square)

    y0_hat = np.dot(T, np.array([tau]).T) + np.dot(X, np.array([beta]).T)
    y1_hat = y0_hat + delta

    t0_hat = np.dot(X, np.array([gamma]).T)
    t1_hat = t0_hat + alpha

    L =     0.5 * coeff1 * np.exp( - coeff2 * (Y - y0_hat)**2) * (((np.exp(t0_hat))**T)/(1 + np.exp(t0_hat)))
    L = L + 0.5 * coeff1 * np.exp( - coeff2 * (Y - y1_hat)**2) * (((np.exp(t1_hat))**T)/(1 + np.exp(t1_hat)))

    LL = np.sum(np.log(L+ 0.000001))
    return -LL

def get_R_2_Y(sigmas_square_hat, sigmas_Y):
    return (np.abs(1 - (sigmas_square_hat/sigmas_Y)))

def get_R_2_T(gamma, sigmas_X, alpha, delta):
    #gamma is (m_X,), sigma_X is (m_X by m_X)
    num = np.dot(np.dot(np.array([gamma]), sigmas_X), np.array([gamma]).T) + (alpha**2)/4
    denom = num + (np.pi**2)/3
    return (np.abs(num[0][0])/denom[0][0])

def run_imbens(X_df, T_df, Y_df, alpha, delta):
    n   = X_df.shape[0]
    m_X = X_df.shape[1] 
    m_U = 1
    
    res = minimize(imbens_obj, 
                   list(np.ones((m_X + m_U + m_X + 1))), 
                   args = {"X": X_df.values, 
                           "T": T_df.values, 
                           "Y": Y_df.values,
                           "alpha": np.array([alpha]), 
                           "delta": np.array([delta])},
                   method='BFGS', 
                   options={'gtol': 1e-5, 'maxiter': 100})
   
    MLE = res.x
    
    res = {}
    res["gamma_hat"]        = np.array(MLE[0:m_X])
    res["alpha"]            = np.array([alpha])

    res["tau_hat"]          = np.array(MLE[m_X:(m_X + 1)])
    res["beta_hat"]         = np.array(MLE[(m_X + 1):(m_X + 1 + m_X)])
    res["delta"]            = np.array([delta])
    res["sigma_square_hat"] = MLE[-1]**2
    
    #sigmas_Y is scalar
    sigmas_Y = np.sum((Y_df.values - np.mean(Y_df.values))**2)/Y_df.shape[0]
    res["R_2_Y"] = get_R_2_Y(res["sigma_square_hat"], sigmas_Y)
    
    #sigmas_X is matrix
    sigmas_X = np.cov(X_df.T)
    res["R_2_T"] = get_R_2_T(res["gamma_hat"], sigmas_X, res["alpha"], res["delta"]) 

    return res

# contour 2d
def binary_search_imbens(X_df, T_df, Y_df, 
                         tau,
                         version, fix_var_value, 
                         interval_lb = 0, interval_ub = 5, 
                         max_iter = 20, precision = 0.01, epsilon = 10**(-5), 
                         verbose = False):
    
    #X_df: n by w_n observed covars
    #T_df: n by 1 binary 0,1 treatment status
    #Y_df: n by 1 outcome 
    
    #version: pick from "alpha", "delta". indicate which variable to search, the other will need to be provided
    #fix_var_value: the value for the other varible that need to be fixed
    
    #interval_lb, interval_ub: 2 numbers control the range of to search
    #max_iter: max binary search depth
    
    #tau: the idea effect size we are after
    #precision and epsilon: algorithm stop if tau_hat is close to tau: +- precision * tau + epsilon
    
    #verbos: print debug info 
    
    #perform binary search in the specified
    #assume that effect size is monotonic function of delta size when alpha is fixed, as delta increases, effect size decreases 
    #assume that effect size is monotonic function of alpha size when alpha is fixed, as alpha increases, effect size decreases

    iter_counter = 0
    lb = interval_lb
    ub = interval_ub
    converge = False 
    
    if version == "alpha":
        print(f"searching for alpha while fix delta: {fix_var_value}")
        fix_var = "delta"
        search_var = "alpha"
    elif version == "delta":
        print(f"searching for delta while fix alpha: {fix_var_value}")
        fix_var = "alpha"
        search_var = "delta"
    else:
        print("wrong argument, version has to be set to alpha or delta")
        return

    
    while iter_counter < max_iter and not converge:
        iter_counter = iter_counter + 1
        if verbose:
            print(f"working on interval: {lb}  to {ub}")

        val_test = (lb + ub)/2
        if version == "alpha":
            mdl = run_imbens(X_df, T_df, Y_df, 
                             alpha = val_test, delta = fix_var_value)
        else:
            mdl = run_imbens(X_df, T_df, Y_df, 
                             alpha = fix_var_value, delta = val_test)
            
        
        tau_hat = mdl["tau_hat"][0]
        if verbose:
            print(f"estimated tau hat under {search_var} = {val_test}: {tau_hat}")
        
        #test for convergence 
        if (np.abs(tau_hat - tau) < (precision * np.abs(tau) + epsilon)):
            converge = True 
            break
        #prepare for next iteration of binary search
        if mdl["tau_hat"][0] >  tau:
            lb = val_test
            ub = ub

        if mdl["tau_hat"][0] <= tau:
            lb = lb
            ub = val_test
    return (mdl, converge)

def contour_2d_search(target_tau,
                      X_df, T_df, Y_df,
                      #plot interval
                      min_alpha = 0, max_alpha = 2, num_alpha = 20,
                      min_delta = 0, max_delta = 2, num_delta = 20,
                      outer_verbose = True,

                      # search interval
                      smin_alpha = 0, smax_alpha = 10,
                      smin_delta = 0, smax_delta = 5,

                      precision = 0.01, epsilon = 10**(-5), max_iter = 20, 
                      inner_verbose = False):
    
    #target_tau: a number indicate the desired tau you want to search for 
    #min_alpha, max_alpha, num_alpha: three numbers that controls a sequnce of alphas equally spaced 
    #min_delta, max_delta, num_delta: three numbers that controls a sequnce of deltas equally spaced 
     
    #smin_alpha, smax_alpha: controls for the binary search space for alpha 
    #smin_delta, smax_delta: controls for the binary search space for delta
    
    #precision: a number, algorithm stop if tau_hat is close to tau: +- precision * tau + epsilon
    #epsilon: a number indicates the tolerance on target_tau-estimated tau. mostly for when target_tau is set to 0
    #max_iter: a number indicates the maximum number of binary search iteration
    
    #under each alpha, we will do a binary search on delta in smin_delta and smax_delta
    #then the role is reversed, we fix delta and search for alpha in smin_alpha and smax_alpha
    #converged results are concatinated and returned as a pandas dataframe

    result_df = []
    if outer_verbose:
        print("get a baseline performance when alpha and delta are both 0")
    mdl_0 = run_imbens(X_df, T_df, Y_df, alpha = 0, delta = 0)
    
    if outer_verbose:
        print("working on fix alpha, searching delta")
    for alpha in tqdm(np.linspace(min_alpha, max_alpha, num_alpha)):

        mdl,converge = binary_search_imbens(X_df, T_df, Y_df, 
                                            tau = target_tau, 
                                            version = "delta", fix_var_value = alpha, 
                                            interval_lb  = smin_delta, interval_ub = smax_delta, 
                                            max_iter = max_iter, precision = precision, epsilon = epsilon, 
                                            verbose = inner_verbose)

        if converge:  
            if outer_verbose:
                print(f"converged")
            mdl["R_2_Y_par"] = (mdl_0["sigma_square_hat"] - mdl["sigma_square_hat"])/mdl_0["sigma_square_hat"]
            mdl["R_2_T_par"] = (mdl["R_2_T"] - mdl_0["R_2_T"])/(1 - mdl_0["R_2_T"])
            

            mdl_df = pd.DataFrame(np.array([[alpha, mdl["delta"][0], 
                                             mdl["R_2_T_par"], mdl["R_2_Y_par"], mdl["tau_hat"][0]]]),
                                  columns = ["alpha", "delta", "R_2_T_par", "R_2_Y_par", "tau_hat"])
            result_df = result_df + [mdl_df] 
            
        if outer_verbose and not converge:
            print(f"not converge after {max_iter}")
        
    if outer_verbose:
        print("working on fix delta, searching alpha")

    for delta in tqdm(np.linspace(min_delta, max_delta, num_delta)):
        mdl,converge = binary_search_imbens(X_df, T_df, Y_df, 
                                            tau = target_tau,
                                            version = "alpha", fix_var_value = delta, 
                                            interval_lb  = smin_alpha, interval_ub = smax_alpha, 
                                            max_iter = max_iter, precision = precision, epsilon = epsilon, 
                                            verbose = inner_verbose)

        if converge:
            if outer_verbose:
                print(f"converged")
            mdl["R_2_Y_par"] = (mdl_0["sigma_square_hat"] - mdl["sigma_square_hat"])/mdl_0["sigma_square_hat"]
            mdl["R_2_T_par"] = (mdl["R_2_T"] - mdl_0["R_2_T"])/(1 - mdl_0["R_2_T"])
            

            mdl_df = pd.DataFrame(np.array([[mdl["alpha"][0], delta, 
                                             mdl["R_2_T_par"], mdl["R_2_Y_par"], mdl["tau_hat"][0]]]),
                                  columns = ["alpha", "delta", "R_2_T_par", "R_2_Y_par", "tau_hat"])

            result_df = result_df + [mdl_df]
            
        if outer_verbose and not converge:
            print(f"Warning not converged after {max_iter} iterations of binary search")
            print(f"consider relax the precision or expand the seach space: decreasing smin and increasing smax")


            
    result_df = pd.concat(result_df) 
    result_df["R_2_T_par"] = np.abs(result_df["R_2_T_par"])
    result_df["R_2_Y_par"] = np.abs(result_df["R_2_Y_par"])
    return result_df

def contour_2d_omit_vars(X_df, T_df, Y_df, verbose = False):
    # this function calculate the R_2_T and R_2_Y which hide each observed covariate
    # return a dataframe that has observed covar names on the row, several important ploting metric on the column
    omit_vars = []
    omit_vars_df = []
    
    mdl_0 = run_imbens(X_df, T_df, Y_df, alpha = 0, delta = 0)
    const_col = np.argsort(np.std(X_df.values, axis = 0))[0]
    
    for x_i in range(X_df.shape[1]):
        if x_i != const_col:
            if verbose:
                print(f"working on {X_df.columns[x_i]}")

            X_res_df = X_df.drop(X_df.columns[x_i], axis = 1)
            mdl = run_imbens(X_res_df, T_df, Y_df, alpha = 0, delta = 0)
            mdl["R_2_Y_par"] = np.abs((mdl_0["sigma_square_hat"] - mdl["sigma_square_hat"])/mdl_0["sigma_square_hat"])
            mdl["R_2_T_par"] = np.abs((mdl["R_2_T"] - mdl_0["R_2_T"])/(1 - mdl_0["R_2_T"]))

            mdl_df = pd.DataFrame(np.array([[0, 0, 
                                             mdl["R_2_T_par"], mdl["R_2_Y_par"], mdl["tau_hat"][0]]]),
                                  columns = ["alpha", "delta", "R_2_T_par", "R_2_Y_par", "tau_hat"])
            omit_vars_df = omit_vars_df + [mdl_df] 
            omit_vars    = omit_vars + [X_df.columns[x_i]]
    omit_vars_df = pd.concat(omit_vars_df, axis = 0)
    omit_vars_df.index = omit_vars
    return(omit_vars_df)

def contour_2d_plot(result_2d_df, 
                    figsize = (10,5), title = "contour plot",
                    smooth = True, kx = 2, 
                    show_covars = False, omit_vars_df = None, 
                    colors = ["orange","gold","royalblue", "forestgreen"],
                    save_as = None, dpi = 300):
    
    #result_2d_df: should contain column: "alpha", "delta", "R_2_T_par", "R_2_Y_par". 
    #rows are different configuration s.t tau is close to desired value
    #figuresize: a tuple of two numbers, indicate the width and height of the figure
    #title: a string, the shared title of the subplots
    #smooth: boolean, indicates if should plot smoothed line or plot raw dots
    #inter_k: a number, indicates the degree of polynomial to fit the dots
    #show_covars: a indicator variable to see if we should plot each observed covar as a dot
    #omit_vars_df: a datafram includes the result from contour_2d_omit_vars
    #fmts: different markers when ploting omit observed covars
    #save_as: a file path, indicates where to save the result figure. 
    #dpi: a number, controls the resolution of the saved result figure
    
    fig = plt.figure(figsize=figsize)
    plt.rcParams['grid.color'] = "silver"

    if not smooth:
        ax1 = fig.add_subplot(121)
        ax1.scatter(x= result_2d_df["alpha"].values, y = result_2d_df["delta"].values, s = 2)
        ax1.set_xlabel(r'$\alpha$', fontweight ='bold')
        ax1.set_ylabel(r'$\delta$', fontweight ='bold')

        ax2 = fig.add_subplot(122)
        ax2.scatter(x= result_2d_df["R_2_T_par"].values, y = result_2d_df["R_2_Y_par"].values, s = 2)
        ax2.set_xlabel(r'$R^2_{T,par}$', fontweight ='bold')
        ax2.set_ylabel(r'$R^2_{Y,par}$', fontweight ='bold')
    
    else:
        #smoothed version
        ax1 = fig.add_subplot(121)
        x = result_2d_df["alpha"].values
        y = result_2d_df["delta"].values
        spl = interpolate.make_interp_spline(x[np.argsort(x)], y[np.argsort(x)], k = kx)
        x = np.linspace(x.min(), x.max(), 500)
        y = spl(x)
        ax1.plot(x, y)
        ax1.set_xlabel(r'$\alpha$', fontweight ='bold')
        ax1.set_ylabel(r'$\delta$', fontweight ='bold')

        ax2 = fig.add_subplot(122)
        x = result_2d_df["R_2_T_par"].values
        y = result_2d_df["R_2_Y_par"].values
        spl = interpolate.make_interp_spline(x[np.argsort(x)], y[np.argsort(x)], k = kx)
        x = np.linspace(min(0, np.min(omit_vars_df["R_2_T_par"])), 
                        max(1, np.max(omit_vars_df["R_2_T_par"])), 500)
        y = spl(x)
        ax2.plot(x, y)
        ax2.set_xlim([0,1])
        ax2.set_ylim([0,1])
        ax2.set_xlabel(r'$R^2_{T,par}$', fontweight ='bold')
        ax2.set_ylabel(r'$R^2_{Y,par}$', fontweight ='bold')
        
    
    # add dots from observed covars
    if show_covars:
        for covar_i in range(len(omit_vars_df.index)):
            covar = omit_vars_df.index[covar_i]
            ax2.scatter(omit_vars_df.loc[covar]["R_2_T_par"], 
                        omit_vars_df.loc[covar]["R_2_Y_par"], 
                        #fmts[covar_i], 
                        label=covar, 
                        c=colors[covar_i])
        ax2.legend()
    
    
    ax2.set_ylim(-0.01,  1)
    ax2.set_xlim(-0.01, 1)
    plt.subplots_adjust(wspace=0.4)
    fig.suptitle(title)
    
    if save_as:
        fig.savefig(save_as, dpi=dpi, 
                    bbox_inches="tight", pad_inches=0.5,
                    facecolor='auto', edgecolor='auto')
    plt.show()
    
#contour 3d
def contour_3d_search(X_df, T_df, Y_df,
                      min_alpha, max_alpha, num_alpha,
                      min_delta, max_delta, num_delta):
    
    mdl_0 = run_imbens(X_df, T_df, Y_df, alpha = 0, delta = 0)
    
    mdls = {}
    result_df = []
    for alpha in tqdm(np.linspace(min_alpha, max_alpha, num_alpha)):
        for delta in np.linspace(min_delta, max_delta, num_delta):
            mdl = run_imbens(X_df, T_df, Y_df, alpha = alpha, delta = delta)
            mdl["R_2_Y_par"] = np.abs(mdl_0["sigma_square_hat"] - mdl["sigma_square_hat"])/mdl_0["sigma_square_hat"]
            mdl["R_2_T_par"] = np.abs(mdl["R_2_T"] - mdl_0["R_2_T"])/(1 - mdl_0["R_2_T"])
            mdls[f"alpha:{alpha}_delta:{delta}"] = mdl

            mdl_df = pd.DataFrame(np.array([[alpha, delta, mdl["R_2_T_par"], mdl["R_2_Y_par"], mdl["tau_hat"][0]]]),
                                  columns = ["alpha", "delta", "R_2_T_par", "R_2_Y_par", "tau_hat"])

            result_df = result_df + [mdl_df]       
    result_df = pd.concat(result_df)
    return result_df

def contour_3d_plot(result_df, figsize = (10,5), 
                    title = "3D contour plot",
                    true_tau_lim = True,
                    alpha_kx = 3, delta_ky = 3, RT_kx = 3, RY_ky = 3,
                    save_as = None, dpi = 300):
             
    num_alpha  = len(np.unique(result_df["alpha"]))
    num_delta  = len(np.unique(result_df["delta"]))
    alpha_plot = result_df["alpha"].values.reshape(num_alpha, num_delta)
    delta_plot = result_df["delta"].values.reshape(num_alpha, num_delta)

    RT_plot    = result_df["R_2_T_par"].values.reshape(num_alpha, num_delta)
    RY_plot    = result_df["R_2_Y_par"].values.reshape(num_alpha, num_delta)

    tau_plot   = result_df["tau_hat"].values.reshape(num_alpha, num_delta)

    
   
    tck1             = interpolate.SmoothBivariateSpline(alpha_plot.flatten(), delta_plot.flatten(), tau_plot.flatten(), 
                                                         kx = alpha_kx, ky = delta_ky) 
    new_alpha_plot, new_delta_plot = np.mgrid[(np.min(alpha_plot)):(np.max(alpha_plot)):100j, 
                                              (np.min(delta_plot)):(np.max(delta_plot)):100j]
   
    new_tau_plot_1   = tck1.ev(new_alpha_plot.flatten(), new_delta_plot.flatten()).reshape([100,100])
    
    
   
    new_RT_plot, new_RY_plot = np.mgrid[0:(np.max(RT_plot)):100j, 
                                        0:(np.max(RY_plot)):100j]
   
        
    tck2             = interpolate.SmoothBivariateSpline(RT_plot.flatten(), RY_plot.flatten(), tau_plot.flatten(), 
                                                         kx =  RT_kx, ky = RY_ky)      
        
    new_tau_plot_2   = tck2.ev(new_RT_plot.flatten(), new_RY_plot.flatten()).reshape([100,100])

    

    fig = plt.figure(figsize=figsize)
    plt.rcParams['grid.color'] = "silver"
    ax1 = fig.add_subplot(121,projection='3d')
    ax1.plot_surface(new_alpha_plot, new_delta_plot, new_tau_plot_1, 
                     cmap='cividis', 
                     rstride=1, cstride=1, alpha=1, antialiased=True, zorder = 1)

    plt.title(r"$\hat{\tau}(\alpha,\delta)$ landscape on $\alpha$, $\delta$")
    ax1.set_xlabel(r'$\alpha$', fontweight ='bold')
    ax1.set_ylabel(r'$\delta$', fontweight ='bold')
    ax1.set_zlabel(r'$\hat{\tau}(\alpha,\delta)$', fontweight ='bold', labelpad = 10)
       
    
    ax2 = fig.add_subplot(122,projection='3d')
    ax2.plot_surface(new_RT_plot, new_RY_plot, new_tau_plot_2, 
                     cmap='cividis', 
                     rstride=1, cstride=1, alpha=1, antialiased=True)

    plt.title(r"$\hat{\tau}(\alpha,\delta)$ landscape on $R^2_{T,par}$, $R^2_{Y,par}$")
    ax2.set_xlabel(r'$R^2_{T,par}$', fontweight ='bold')
    ax2.set_ylabel(r'$R^2_{Y,par}$', fontweight ='bold')
    ax2.set_zlabel(r'$\hat{\tau}(\alpha,\delta)$', fontweight ='bold', labelpad = 10, zorder = 1)
    

    # Get rid of colored axes planes
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.axes.set_facecolor('w')

    # Now set color to white (or whatever is "invisible")
    ax1.xaxis.pane.set_edgecolor('grey')
    ax1.yaxis.pane.set_edgecolor('grey')
    ax1.zaxis.pane.set_edgecolor('grey')
    
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    ax2.xaxis.axes.set_facecolor('w')

    ax2.xaxis.pane.set_edgecolor('grey')
    ax2.yaxis.pane.set_edgecolor('grey')
    ax2.zaxis.pane.set_edgecolor('grey')
    
    
    #same Z
    if true_tau_lim:
        min_z = np.min(tau_plot)
        max_z = np.max(tau_plot)
    else:
        min_z = min(np.min(new_tau_plot_1), np.min(new_tau_plot_2))
        max_z = max(np.max(new_tau_plot_1), np.max(new_tau_plot_2))
        
    ax1.set_zlim(min_z-1, max_z)
    ax2.set_zlim(min_z-1, max_z)
    
    #R^2 plot should have origin (0,0)
    ax1.invert_xaxis()
    ax2.invert_xaxis()
    
    cset =  ax1.contourf(new_alpha_plot, new_delta_plot, (new_tau_plot_1 - np.min(new_tau_plot_1) ),
                   zdir ='z',
                   offset = np.min(tau_plot),cmap ='bone')
  

    cset =  ax2.contourf(new_RT_plot, new_RY_plot, (new_tau_plot_2 - np.min(new_tau_plot_2) ),
                   zdir ='z',
                   offset = np.min(tau_plot),cmap ='bone')

    # two plot a little seperate
    plt.subplots_adjust(wspace=0.4)

    if save_as:
        fig.savefig(save_as, dpi=dpi, 
                    bbox_inches="tight", pad_inches=0.5,
                    facecolor='auto', edgecolor='auto',)

    plt.show()


