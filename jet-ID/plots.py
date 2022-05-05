import numpy as np
import h5py, pickle, sys, time
from   sklearn           import metrics
from   scipy.spatial     import distance
from   matplotlib        import pylab
from   matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, FixedLocator
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def valid_accuracy(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    return np.sum(y_pred==y_true)/len(y_true)


def LLH_rates(sample, y_true, ECIDS=False):
    LLH_tpr, LLH_fpr = [],[]
    for wp in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']:
        y_class0 = sample[wp][y_true == 0]
        y_class1 = sample[wp][y_true != 0]
        LLH_tpr.append( np.sum(y_class0 == 0)/len(y_class0) )
        LLH_fpr.append( np.sum(y_class1 == 0)/len(y_class1) )
    if ECIDS:
        ECIDS_cut    = -0.337671
        ECIDS_class0 = sample['p_ECIDSResult'][y_true == 0]
        ECIDS_class1 = sample['p_ECIDSResult'][y_true != 0]
        for wp in ['p_LHTight', 'p_LHMedium', 'p_LHLoose']:
            y_class0 = sample[wp][y_true == 0]
            y_class1 = sample[wp][y_true != 0]
            LLH_tpr.append( np.sum( (y_class0 == 0) & (ECIDS_class0 >= ECIDS_cut) )/len(y_class0) )
            LLH_fpr.append( np.sum( (y_class1 == 0) & (ECIDS_class1 >= ECIDS_cut) )/len(y_class1) )
    return LLH_fpr, LLH_tpr


def plot_history(history, output_dir, key='loss'):
    if history == None or len(history.epoch) < 2: return
    plt.figure(figsize=(12,8))
    pylab.grid(True)
    val = plt.plot(np.array(history.epoch)+1, 100*np.array(history.history[key]), label='Training')
    plt.plot(np.array(history.epoch)+1, 100*np.array(history.history['val_'+key]), '--',
             color=val[0].get_color(), label='Testing')
    min_acc = np.floor(100*min( history.history[key]+history.history['val_'+key] ))
    max_acc = np.ceil (100*max( history.history[key]+history.history['val_'+key] ))
    plt.xlim([1, max(history.epoch)+1])
    plt.xticks( np.append(1, np.arange(5, max(history.epoch)+2, step=5)) )
    plt.xlabel('Epochs',fontsize=25)
    plt.ylim( max(70,min_acc),max_acc )
    plt.yticks( np.arange(max(80,min_acc), max_acc+1, step=1) )
    plt.ylabel(key.title()+' (%)',fontsize=25)
    plt.legend(loc='lower right', fontsize=20, numpoints=3)
    file_name = output_dir+'/'+'history.png'
    print('Saving training accuracy history to:', file_name, '\n'); plt.savefig(file_name)


def plot_heatmaps(sample, labels, output_dir):
    n_classes = max(labels)+1
    label_dict = {0:'iso electron', 1:'charge flip'  , 2:'photon conversion'    , 3  :'b/c hadron',
                  4:'light flavor ($\gamma$/e$^\pm$)', 5:'light flavor (hadron)'}
    pt  =     sample['pt']  ;  pt_bins = np.arange(0,81,1)
    eta = abs(sample['eta']); eta_bins = np.arange(0,2.55,0.05)
    extent = [eta_bins[0], eta_bins[-1], pt_bins[0], pt_bins[-1]]
    fig = plt.figure(figsize=(20,10)); axes = plt.gca()
    for n in np.arange(n_classes):
        plt.subplot(2, 3, n+1)
        heatmap = np.histogram2d(eta[labels==n], pt[labels==n], bins=[eta_bins,pt_bins], density=False)[0]
        plt.imshow(heatmap.T, origin='lower', extent=extent, cmap='Blues', interpolation='bilinear', aspect="auto")
        plt.title(label_dict[n]+' ('+format(100*np.sum(labels==n)/len(labels),'.1f')+'%)', fontsize=25)
        if n//3 == 1: plt.xlabel('abs('+'$\eta$'+')', fontsize=25)
        if n %3 == 0: plt.ylabel('$p_t$ (GeV)', fontsize=25)
    fig.subplots_adjust(left=0.05, top=0.95, bottom=0.1, right=0.95, wspace=0.15, hspace=0.25)
    file_name = output_dir+'/'+'heatmap.png'
    print('Saving heatmap plots to:', file_name)
    plt.savefig(file_name); sys.exit()


def var_histogram(sample, labels, weights, bins, output_dir, prefix, var, density=True, separate_norm=False):
    n_classes = max(labels)+1
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    if var == 'pt':
        variable = sample[var]
        tag=''; plt.xlabel(tag+'$p_t$ (GeV)', fontsize=25)
        if bins == None or len(bins[var]) <= 2:
            #bins = [0, 10, 20, 30, 40, 60, 80, 100, 130, 180, 250, 500]
            bins = np.arange(np.min(variable), 101, 1)
        else: bins = bins[var]
        axes.xaxis.set_minor_locator(FixedLocator(bins))
        #plt.xticks(np.arange(0, bins[-1]+100, 100)); pylab.xlim(5, bins[-1]); plt.xlim(0, 100)
        if n_classes == 2: plt.xlim(5, 1000); plt.ylim(1e-5,100); plt.xscale('log'); plt.yscale('log')
        else             : plt.xlim(5, 1000); plt.ylim(1e-5,10) ; plt.xscale('log'); plt.yscale('log')
    if var == 'eta':
        variable = abs(sample[var])
        tag=''; plt.xlabel('abs($\eta$)', fontsize=25)
        if bins == None or len(bins[var]) <= 2:
            #bins = [0, 0.1, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37, 2.47]
            step = 0.05; bins = np.arange(0, 2.5+step, step)
        else: bins = bins[var]
        axes.xaxis.set_minor_locator(FixedLocator(bins)) #axes.xaxis.set_minor_locator(AutoMinorLocator(10))
        plt.xticks(np.arange(0, 2.6, 0.1))
        #plt.xticks(bins, [str(n) for n in bins]); plt.xticks(bins, [format(n,'.1f') for n in bins])
        pylab.xlim(0, 2.5)
    bins[-1] = max(bins[-1], max(variable)+1e-3)
    label_dict = {0:'iso electron', 1:'charge flip'  , 2:'photon conversion'    , 3  :'b/c hadron',
                  4:'light flavor ($\gamma$/e$^\pm$)', 5:'light flavor (hadron)'}
    color_dict = {0:'tab:blue'    , 1:'tab:orange'   , 2:'tab:green'            , 3  :'tab:red'   ,
                  4:'tab:purple'                     , 5:'tab:brown'            }
    if n_classes == 2: label_dict[1] = 'background'
    if n_classes != 2 and var == 'eta': separate_norm = True
    if np.all(weights) == None: weights = np.ones(len(variable))
    h = np.zeros((len(bins)-1,n_classes))
    for n in np.arange(n_classes):
        class_values  = variable[labels==n]; class_weights = weights[labels==n]
        class_weights = 100*class_weights/(np.sum(class_weights) if separate_norm else len(variable))
        if density:
            indices        = np.searchsorted(bins, class_values, side='right')
            class_weights /= np.take(np.diff(bins), np.minimum(indices, len(bins)-1)-1)
        label  = 'class '+str(n)+': '+label_dict[n]
        h[:,n] = pylab.hist(class_values, bins, histtype='step', weights=class_weights, color=color_dict[n],
                            label=label+' ('+format(100*len(class_values)/len(variable),'.1f')+'%)', lw=2)[0]
    plt.ylabel('Distribution density (%)' if density else 'Distribution (%)', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=12)
    plt.legend(loc='upper right', fontsize=16 if n_classes==2 else 14)
    file_name = output_dir+'/'+str(var)+'_'+prefix+'.png'
    print('Saving', prefix, 'sample', format(var,'3s'), 'distributions to:', file_name)
    plt.savefig(file_name)


def plot_distributions_DG(sample, y_true, y_prob, output_dir, separation=False, bkg='bkg'):
    #variable = sample['rljet_topTag_DNN19_qqb_score']
    variable = y_prob
    label_dict = {0:'iso electron', 1:'charge flip'  , 2:'photon conversion'    ,   3  :'heavy flavor (b/c)',
                  4:'light flavor (e$^\pm$/$\gamma$)', 5:'light flavor (hadron)', 'bkg':'background'}
    color_dict = {0:'tab:blue'    , 1:'tab:orange'   , 2:'tab:green'            ,   3  :'tab:red'   ,
                  4:'tab:purple'                     , 5:'tab:brown'            , 'bkg':'tab:orange'}
    if separation: label_dict.pop('bkg')
    else: label_dict={0:r'Signal', 1:'QCD'}; color_dict={0:'tab:orange', 1:'tab:blue'}
    n_classes = len(label_dict)
    def logit(x, delta=1e-16):
        x = np.clip(np.float64(x), delta, 1-delta)
        return np.log10(x) - np.log10(1-x)
    def inverse_logit(x):
        return 1/(1+10**(-x))
    def print_JSD(P, Q, idx, color, text):
        plt.text(0.945, 1.01-3*idx/100, 'JSD$_{0,\!'+text+'}$:',
                 {'color':'black', 'fontsize':10}, va='center', ha='right', transform=axes.transAxes)
        plt.text(0.990, 1.01-3*idx/100, format(distance.jensenshannon(P, Q), '.3f'),
                 {'color':color  , 'fontsize':10}, va='center', ha='right', transform=axes.transAxes)
    def class_histo(y_true, variable, bins, colors):
        h = np.full((len(bins)-1,n_classes), 0.)
        from utils import make_labels; class_labels = make_labels(sample, n_classes)
        for n in np.arange(n_classes):
            class_var = variable[class_labels==n]
            class_weights  =  sample['weights'][class_labels==n]
            #class_weights *= 100/(np.sum(sample['weights']))
            class_weights *= 100/np.sum(class_weights)
            pylab.hist(class_var, bins=bins, label=label_dict[n], histtype='step',
                       weights=class_weights, log=True, color=colors[n], lw=2, alpha=1)[0]

            #class_weights = len(class_probs)*[100/len(y_true)]
            #class_weights = len(class_var)*[100/len(class_var)]
            #h[:,n] = pylab.hist(class_var, bins=bins, label=label_dict[n]+' (unweighted)', histtype='step',
            #                    weights=class_weights, log=True, color=colors[n], lw=2, alpha=0.5)[0]

        if n_classes == 2: colors = len(colors)*['black']
        if False:
            for n in np.arange(1, n_classes):
                new_y_true = y_true[np.logical_or(y_true==0, class_labels==n)]
                new_y_prob = y_prob[np.logical_or(y_true==0, class_labels==n)]
                fpr, tpr, threshold = metrics.roc_curve(new_y_true, new_y_prob, pos_label=0)
                sig_ratio = np.sum(y_true==0)/len(new_y_true)
                #sig_ratio = np.sum(sample['weights'][class_labels==0])/np.sum(sample['weights'])
                max_index = np.argmax(sig_ratio*tpr + (1-fpr)*(1-sig_ratio))
                axes.axvline(threshold[max_index], ymin=0, ymax=1, ls='--', lw=1, color=colors[n])
        #for n in np.arange(1, n_classes): print_JSD(h[:,0], h[:,n], n, colors[n], str(n))
        #if n_classes > 2: print_JSD(h[:,0], np.sum(h[:,1:], axis=1), n_classes, 'black', '\mathrm{bkg}')
    plt.figure(figsize=(12,16))
    plt.subplot(2, 1, 1); pylab.grid(True); axes = plt.gca()
    pylab.xlim(0,100); pylab.ylim(1e-5 if n_classes>2 else 1e-3, 1e2)
    plt.xticks(np.arange(0,101,step=10))
    #pylab.xlim(0,10); pylab.ylim(1e-2 if n_classes>2 else 1e-2, 1e2)
    #plt.xticks(np.arange(0,11,step=1))
    bin_step = 0.5; bins = np.arange(0, 100+bin_step, bin_step)
    class_histo(y_true, 100*variable, bins, color_dict)
    plt.xlabel('$p_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('Distribution (%)', fontsize=25)
    axes.tick_params(axis='both', which='major', labelsize=12)

    handles, labels = plt.gca().get_legend_handles_labels()
    #order = [2,3,0,1]
    order = [0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper right',
               fontsize=16 if n_classes==2 else 14, ncol=1)

    plt.subplot(2, 1, 2); pylab.grid(True); axes = plt.gca()
    x_min=-6; x_max=5; pylab.xlim(x_min, x_max); pylab.ylim(1e-4 if n_classes>2 else 1e-3, 1e1)
    #x_min=-7; x_max=4; pylab.xlim(x_min, x_max); pylab.ylim(1e-4 if n_classes>2 else 1e-3, 1e1)
    pos  =                   [  10**float(n)      for n in np.arange(x_min,0)       ]
    pos += [0.5]           + [1-10**float(n)      for n in np.arange(-1,-x_max-1,-1)]
    lab  =                   ['$10^{'+str(n)+'}$' for n in np.arange(x_min+2,0)     ]
    lab += [1,10,50,90,99] + ['99.'+n*'9'         for n in np.arange(1,x_max-1)     ]
    #x_min=-10; x_max=-1; pylab.xlim(x_min, x_max); pylab.ylim(1e-2 if n_classes>2 else 1e-4, 1e2)
    #pos  =                   [  10**float(n)      for n in np.arange(x_min,0)       ]
    #lab  =                   ['$10^{'+str(n)+'}$' for n in np.arange(x_min+2,0)     ] + [1,10]
    #lab += ['0.50   '] + ['$1\!-\!10^{'+str(n)+'}$' for n in np.arange(-1,-x_max-1,-1)]
    plt.xticks(logit(np.array(pos)), lab, rotation=15)
    bin_step = 0.1; bins = np.arange(x_min-1, x_max+1, bin_step)
    #y_prob = logit(y_prob)
    class_histo(y_true, logit(variable), bins, color_dict)
    plt.xlabel('$p_{\operatorname{sig}}$ (%)', fontsize=25)
    plt.ylabel('Distribution (%)', fontsize=25)
    location = 'upper right' if n_classes==2 else 'upper left'
    axes.tick_params(axis='both', which='major', labelsize=12)

    handles, labels = plt.gca().get_legend_handles_labels()
    #order = [2,3,0,1]
    order = [0,1]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=location,
               fontsize=16 if n_classes==2 else 14, ncol=1)

    plt.subplots_adjust(top=0.95, bottom=0.1, hspace=0.2)
    file_name = output_dir+'/distributions.png'
    print('Saving test sample distributions to:', file_name); plt.savefig(file_name)


def plot_weights(sample, y_true, probs, output_dir, n_bins=600):
    labels = [r'$t\bar{t}$ jets', 'QCD jets']; colors = ['tab:orange', 'tab:blue']
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    for n in [0,1]:
        variable = sample['rljet_topTag_DNN19_qqb_score'][y_true==n]
        #print( np.min(variable), np.max(variable), np.sum(variable==-1000) )
        #variable = sample['weights'][y_true==n]
        #variable = probs[y_true==n]
        min_val  = np.min(variable)
        max_val  = np.max(variable)
        bin_width = (max_val-min_val)/n_bins
        bins      = list(np.arange(min_val, max_val, bin_width)) + [max_val]
        pylab.hist(variable, bins, histtype='step', weights=None,
                   log=True, label=labels[n], color=colors[n], lw=2)
    #plt.xlim([0,10000])
    plt.xlabel('Weight value', fontsize=24)
    plt.ylabel('Distribution  (entries)', fontsize=24)
    plt.legend(loc='upper left', fontsize=18)
    file_name = output_dir+'/'+'weights.png'
    print('Saving weights to:', file_name); plt.savefig(file_name)


def plot_ROC_curves(sample, y_true, y_prob, output_dir, ROC_type, ECIDS, ROC_values=None):
    #LLH_fpr, LLH_tpr = LLH_rates(sample, y_true, ECIDS)
    '''
    if ROC_values != None:
        index = output_dir.split('_')[-1]
        index = ROC_values[0].shape[1]-1 if index == 'bkg' else int(index)
        fpr_full, tpr_full = ROC_values[0][:,index], ROC_values[0][:,0]
        fpr     , tpr      = ROC_values[1][:,index], ROC_values[1][:,0]
    else:
        toptag  = sample['rljet_topTag_DNN19_qqb_score']
        weights = sample['weights']
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_prob, pos_label=0, sample_weight=weights)
        fpr_tag, tpr_tag, threshold = metrics.roc_curve(y_true, toptag, pos_label=0, sample_weight=weights, drop_intermediate=False)
    signal_ratio        = np.sum(y_true==0)/len(y_true)
    accuracy            = tpr*signal_ratio + (1-fpr)*(1-signal_ratio)
    best_tpr, best_fpr  = tpr[np.argmax(accuracy)], fpr[np.argmax(accuracy)]
    '''
    colors  = ['red', 'blue', 'green', 'red', 'blue', 'green']
    labels  = ['tight'      , 'medium'      , 'loose'       ,
               'tight+ECIDS', 'medium+ECIDS', 'loose+ECIDS']
    markers = 3*['o'] + 3*['D']
    sig_eff, bkg_eff = '$\epsilon_{\operatorname{sig}}$', '$\epsilon_{\operatorname{bkg}}$'
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    if ROC_type == 1:
        #plt.xlim([0.6, 1]); plt.ylim([0.9, 1-1e-4])
        #plt.xticks([0.6, 0.7, 0.8, 0.9, 1], [60, 70, 80, 90, 100])
        #plt.yscale('logit')
        #plt.yticks([0.9, 0.99, 0.999, 0.9999], [90, 99, 99.9, 99.99])
        axes.xaxis.set_minor_locator(AutoMinorLocator(10))
        axes.xaxis.set_minor_formatter(plt.NullFormatter())
        axes.yaxis.set_minor_formatter(plt.NullFormatter())
        plt.xlabel(sig_eff+' (%)', fontsize=25)
        #plt.ylabel('$1\!-\!$'+bkg_eff+' (%)', fontsize=25)
        plt.ylabel('1/'+bkg_eff, fontsize=25)

        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob, pos_label=0, sample_weight=sample['weights'])
        pickle.dump({'fpr':fpr, 'tpr':tpr}, open(output_dir+'/'+'pos_rates.pkl','wb'), protocol=4)
        len_0 = np.sum(fpr==0)
        #plt.text(0.5, 0.19, 'AUC: '+format(metrics.auc(fpr,tpr),'.4f'), color='tab:blue', fontsize=19,
        plt.text(0.5, 0.09, 'AUC: '+format(metrics.auc(fpr,tpr),'.4f'), color='tab:blue', fontsize=19,
                 alpha=1, va='center', ha='center', transform=axes.transAxes)
        #plt.plot(100*tpr, 100*(1-fpr), label='Basic FCN', color='tab:blue', alpha=1, lw=2)
        plt.plot(100*tpr[len_0:], 1/fpr[len_0:], label='Basic FCN', color='tab:blue', alpha=1, lw=2)
        #fpr, tpr, _ = metrics.roc_curve(y_true, y_prob, pos_label=0, sample_weight=None)
        #plt.text(0.5, 0.14, 'AUC: '+format(metrics.auc(fpr,tpr),'.4f'), color='tab:blue', fontsize=19,
        #         alpha=0.5, va='center', ha='center', transform=axes.transAxes)
        #plt.plot(100*tpr, 100*(1-fpr), label='Basic FCN (unweighted)', color='#1f77b4', alpha=0.5, lw=2)

        '''
        DNN_tag  = sample['rljet_topTag_DNN19_qqb_score']
        cut = np.logical_and( np.isfinite(DNN_tag), DNN_tag!=-1000 )
        y_true  = y_true[cut]
        DNN_tag = DNN_tag[cut]
        weights = sample['weights'][cut]
        fpr_tag, tpr_tag, _ = metrics.roc_curve(y_true, DNN_tag, pos_label=0, sample_weight=weights)
        #plt.text(0.5, 0.09, 'AUC: '+format(metrics.auc(fpr_tag,tpr_tag),'.4f'), color='tab:orange', fontsize=19,
        plt.text(0.5, 0.04, 'AUC: '+format(metrics.auc(fpr_tag,tpr_tag),'.4f'), color='tab:orange', fontsize=19,
                 alpha=1, va='center', ha='center', transform=axes.transAxes)
        plt.plot(100*tpr_tag, 100*(1-fpr_tag), label='Top tagger', color='tab:orange', alpha=1, lw=2)
        #fpr_tag, tpr_tag, _ = metrics.roc_curve(y_true, DNN_tag, pos_label=0, sample_weight=None)
        #plt.text(0.5, 0.04, 'AUC: '+format(metrics.auc(fpr_tag,tpr_tag),'.4f'), color='tab:orange', fontsize=19,
        #         alpha=0.5, va='center', ha='center', transform=axes.transAxes)
        #plt.plot(100*tpr_tag, 100*(1-fpr_tag), label='Top tagger (unweighted)', color='tab:orange', alpha=0.5, lw=2)
        '''

        axes.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(loc='lower left', fontsize=15, numpoints=3)
    file_name = output_dir+'/ROC'+str(ROC_type)+'_curve.png'
    print('Saving test sample ROC'+str(ROC_type)+' curve to   :', file_name); plt.savefig(file_name)


def combine_ROC_curves(output_dir, cuts=''):
    import multiprocessing as mp, pickle
    from scipy.interpolate import make_interp_spline
    from utils import NN_weights
    def mp_roc(idx, output_dir, return_dict):
        result_file = output_dir+'/'+'results_'+str(idx)+'.pkl'
        sample, labels, probs = pickle.load(open(result_file, 'rb'))
        #cuts = (sample["p_et_calo"] >= 0) & (sample["p_et_calo"] <= 500)
        if cuts == '': cuts = len(labels)*[True]
        sample, labels, probs = {key:sample[key][cuts] for key in sample}, labels[cuts], probs[cuts]
        fpr, tpr, threshold = metrics.roc_curve(labels, probs[:,0], pos_label=0)
        LLH_fpr, LLH_tpr = LLH_rates(sample, labels)
        print('LOADING VALIDATION RESULTS FROM', result_file)
        return_dict[idx] = fpr, tpr, threshold, LLH_fpr, LLH_tpr
    manager  = mp.Manager(); return_dict = manager.dict()
    idx_list = [1, 2, 3, 4]
    names    = ['no weight', 'flattening', 'match2s', 'match2max']
    colors   = ['tab:blue', 'tab:orange', 'tab:green', 'tab:brown']
    lines    = ['-', '-', '-', '-', '--', '--']
    processes = [mp.Process(target=mp_roc, args=(idx, output_dir, return_dict)) for idx in idx_list]
    for job in processes: job.start()
    for job in processes: job.join()
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    file_name = output_dir+'/'+'ROC_curve.png'
    x_min = []; y_max = []
    for n in np.arange(len(idx_list)):
        fpr, tpr, threshold, LLH_fpr, LLH_tpr = return_dict[idx_list[n]]; len_0  = np.sum(fpr==0)
        x_min += [min(60, 10*np.floor(10*LLH_tpr[0]))]
        y_max += [100*np.ceil(max(1/fpr[np.argwhere(tpr >= x_min[-1]/100)[0]], 1/LLH_fpr[0])/100)]
        plt.plot(100*tpr[len_0:], 1/fpr[len_0:], color=colors[n], label=names[n], linestyle=lines[n], lw=2)
        #label = str(bkg_class) if bkg_class != 0 else 'others'
        #val   = plt.plot(100*tpr[len_0:], 1/fpr[len_0:], label='class 0 vs '+label, lw=2)
        #for LLH in zip(LLH_tpr, LLH_fpr): plt.scatter(100*LLH[0], 1/LLH[1], s=40, marker='o', c=val[0].get_color())
    plt.xlim([min(x_min), 100]); plt.ylim([1, 250])  #plt.ylim([1, max(y_max)])
    axes.xaxis.set_major_locator(MultipleLocator(10))
    axes.yaxis.set_ticks( np.append([1], np.arange(50,300,50)) )
    plt.xlabel('Signal Efficiency (%)',fontsize=25)
    plt.ylabel('1/(Background Efficiency)',fontsize=25); #plt.yscale("log")
    plt.legend(loc='upper right', fontsize=15, numpoints=3)
    plt.savefig(file_name); sys.exit()
    '''
    plt.figure(figsize=(12,8)); pylab.grid(True); axes = plt.gca()
    file_name = output_dir+'/'+'ROC1_curve.png'
    for LLH_tpr in [0.7, 0.8, 0.9]:
        bkg_rej  = [1/return_dict[idx][0][np.argwhere(return_dict[idx][1] >= LLH_tpr)[0]][0]
                    for idx in idx_list]
        #bkg_rej /= np.mean(bkg_rej)
        #n_weights = [NN_weights((5,13), CNN_dict, [200, 200], 2) for idx in idx_list]
        #bkg_rej   = [1e5*bkg_rej[n]/n_weights[n] for n in np.arange(len(idx_list))]
        plt.scatter(idx_list, bkg_rej, s=40, marker='o')
        idx_array    = np.linspace(min(idx_list), max(idx_list), 1000)
        spline       = make_interp_spline(idx_list, bkg_rej, k=2)
        plt.plot(idx_array, spline(idx_array), label=format(100*LLH_tpr,'.0f')+'% sig. eff.', lw=2)
    plt.xlim([min(idx_list)-1, max(idx_list)+1])
    axes.xaxis.set_major_locator(MultipleLocator(1))
    plt.ylim([0, 1500])
    axes.yaxis.set_major_locator(MultipleLocator(100))
    plt.xlabel('Maximum Number of Tracks',fontsize=25)
    plt.ylabel('1/(Background Efficiency)',fontsize=25)
    plt.legend(loc='lower center', fontsize=15, numpoints=3)
    plt.savefig(file_name)
    '''


def cal_images(sample, labels, layers, output_dir, mode='random', scale='free', soft=True):
    import multiprocessing as mp
    def get_image(sample, labels, e_class, key, mode, image_dict):
        start_time = time.time()
        if mode == 'random':
            for counter in np.arange(10000):
                image = abs(sample[key][np.random.choice(np.where(labels==e_class)[0])])
                if np.max(image) !=0: break
        if mode == 'mean': image = np.mean(sample[key][labels==e_class], axis=0)
        if mode == 'std' : image = np.std (sample[key][labels==e_class], axis=0)
        print('plotting layer '+format(key,length+'s')+' for class '+str(e_class), end='', flush=True)
        print(' (', '\b'+format(time.time() - start_time, '2.1f'), '\b'+' s)')
        image_dict[(e_class,key)] = image
    layers    = [layer for layer in layers if layer in sample.keys()]
    n_classes = max(labels)+1; length = str(max(len(n) for n in layers))
    manager   =  mp.Manager(); image_dict = manager.dict()
    processes = [mp.Process(target=get_image, args=(sample, labels, e_class, key, mode, image_dict))
                 for e_class in np.arange(n_classes) for key in layers]
    print('PLOTTING CALORIMETER IMAGES (mode='+mode+', scale='+str(scale)+')')
    for job in processes: job.start()
    for job in processes: job.join()
    file_name = output_dir+'/cal_images.png'
    print('SAVING IMAGES TO:', file_name, '\n')
    fig = plt.figure(figsize=(7,14)) if n_classes == 2 else plt.figure(figsize=(18,14))
    for e_class in np.arange(n_classes):
        if scale == 'class': vmax = max([np.max(image_dict[(e_class,key)]) for key in layers])
        for key in layers:
            image_dict[(e_class,key)] -= min(0,np.min(image_dict[(e_class,key)]))
            #image_dict[(e_class,key)] = abs(image_dict[(e_class,key)])
            if scale == 'layer':
                vmax = max([np.max(image_dict[(e_class,key)]) for e_class in np.arange(n_classes)])
            if scale == 'free':
                vmax = np.max(image_dict[(e_class,key)])
            plot_image(100*image_dict[(e_class,key)], n_classes, e_class, layers, key, 100*vmax, soft)
    wspace = -0.1 if n_classes == 2 else 0.2
    fig.subplots_adjust(left=0.05, top=0.95, bottom=0.05, right=0.95, hspace=0.6, wspace=wspace)
    fig.savefig(file_name); sys.exit()


def plot_image(image, n_classes, e_class, layers, key, vmax, soft=True):
    class_dict = {0:'iso electron',  1:'charge flip' , 2:'photon conversion', 3:'b/c hadron',
                  4:'light flavor ($\gamma$/e$^\pm$)', 5:'light flavor (hadron)'}
    layer_dict = {'em_barrel_Lr0'     :'presampler'            , 'em_barrel_Lr1'  :'EM cal $1^{st}$ layer' ,
                  'em_barrel_Lr1_fine':'EM cal $1^{st}$ layer' , 'em_barrel_Lr2'  :'EM cal $2^{nd}$ layer' ,
                  'em_barrel_Lr3'     :'EM cal $3^{rd}$ layer' , 'tile_barrel_Lr1':'had cal $1^{st}$ layer',
                  'tile_barrel_Lr2'   :'had cal $2^{nd}$ layer', 'tile_barrel_Lr3':'had cal $3^{rd}$ layer'}
    if n_classes ==2: class_dict[1] = 'background'
    e_layer  = layers.index(key)
    n_layers = len(layers)
    plot_idx = n_classes*e_layer + e_class+1
    plt.subplot(n_layers, n_classes, plot_idx)
    #title   = class_dict[e_class]+'\n('+layer_dict[key]+')'
    #title   = layer_dict[key]+'\n('+class_dict[e_class]+')'
    title   = class_dict[e_class]+'\n('+str(key)+')'
    limits  = [-0.13499031, 0.1349903, -0.088, 0.088]
    x_label = '$\phi$'                             if e_layer == n_layers-1 else ''
    x_ticks = [limits[0],-0.05,0.05,limits[1]]     if e_layer == n_layers-1 else []
    y_label = '$\eta$'                             if e_class == 0          else ''
    y_ticks = [limits[2],-0.05,0.0,0.05,limits[3]] if e_class == 0          else []
    plt.title(title,fontweight='normal', fontsize=12)
    plt.xlabel(x_label,fontsize=15); plt.xticks(x_ticks)
    plt.ylabel(y_label,fontsize=15); plt.yticks(y_ticks)
    plt.imshow(np.float32(image), cmap='Reds', interpolation='bilinear' if soft else None,
               extent=limits, vmax=1 if np.max(image)==0 else vmax) #norm=colors.LogNorm(1e-3,vmax))
    plt.colorbar(pad=0.02)


def plot_vertex(sample):
    bins = np.arange(0,50,1)
    fig = plt.figure(figsize=(12,8))
    pylab.xlim(-0.5,10.5)
    plt.xticks (np.arange(0,11,1))
    pylab.ylim(0,100)
    plt.xlabel('Track vertex value', fontsize=25)
    plt.ylabel('Distribution (%)', fontsize=25)
    weights = len(sample)*[100/len(sample)]
    pylab.hist(sample, bins=bins, weights=weights, histtype='bar', align='left', rwidth=0.5, lw=2)
    file_name = 'outputs/tracks_vertex.png'
    print('Printing:', file_name)
    plt.savefig(file_name)


def plot_scalars(sample, sample_trans, variable):
    bins = np.arange(-1,1,0.01)
    fig = plt.figure(figsize=(18,8))
    plt.subplot(1,2,1)
    pylab.xlim(-1,1)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Number of Entries')
    #pylab.hist(sample_trans[variable], bins=bins, histtype='step', density=True)
    pylab.hist(sample      [variable], bins=bins, histtype='step', density=False)
    plt.subplot(1,2,2)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Number of Entries')
    pylab.hist(sample_trans[variable], bins=bins)
    file_name = 'outputs/plots/scalars/'+variable+'.png'
    print('Printing:', file_name)
    plt.savefig(file_name)


def plot_tracks(tracks, labels, variable):
    tracks_var = {'efrac':{'idx':0, 'mean_lim':( 0,      3), 'max_lim':(0,    2), 'diff_lim':(0,    1)},
                  'deta' :{'idx':1, 'mean_lim':( 0, 0.0005), 'max_lim':(0, 0.03), 'diff_lim':(0, 0.04)},
                  'dphi' :{'idx':2, 'mean_lim':( 0,  0.001), 'max_lim':(0,  0.1), 'diff_lim':(0, 0.05)},
                  'd0'   :{'idx':3, 'mean_lim':( 0,    0.2), 'max_lim':(0,  0.1), 'diff_lim':(0,  0.3)},
                  'z0'   :{'idx':4, 'mean_lim':( 0,    0.5), 'max_lim':(0,  0.3), 'diff_lim':(0,   10)}}
    classes    = np.arange(max(labels)+1)
    n_e        = np.arange(len(labels)  )
    n_tracks   = np.sum(abs(tracks), axis=2)
    n_tracks   = np.array([len(np.where(n_tracks[n,:]!=0)[0]) for n in n_e])
    var        = tracks[..., tracks_var[variable]['idx']]
    var_mean   = np.array([np.mean(    var[n,:n_tracks[n]])  if n_tracks[n]!=0 else None for n in n_e])
    var_max    = np.array([np.max (abs(var[n,:n_tracks[n]])) if n_tracks[n]!=0 else None for n in n_e])
    var_diff   = np.array([np.mean(np.diff(np.sort(var[n,:n_tracks[n]])))
                           if n_tracks[n]>=2 else None for n in n_e])
    var_diff   = np.array([(np.max(var[n,:n_tracks[n]]) - np.min(var[n,:n_tracks[n]]))/(n_tracks[n]-1)
                           if n_tracks[n]>=2 else None for n in n_e])
    var_mean   = [var_mean[np.logical_and(labels==n, var_mean!=None)] for n in classes]
    var_max    = [var_max [np.logical_and(labels==n, var_max !=None)] for n in classes]
    var_diff   = [var_diff[np.logical_and(labels==n, var_diff!=None)] for n in classes]
    n_tracks   = [n_tracks[labels==n                                ] for n in classes]
    trk_mean   = [np.mean(n_tracks[n])                                for n in classes]
    fig  = plt.figure(figsize=(18,7))
    xlim = (0, 15)
    bins = np.arange(xlim[0], xlim[1]+2, 1)
    for n in [1,2]:
        plt.subplot(1,2,n); axes = plt.gca()
        plt.xlim(xlim)
        plt.xlabel('Number of tracks'      , fontsize=20)
        plt.xticks( np.arange(xlim[0],xlim[1]+1,1) )
        plt.ylabel('Normalized entries (%)', fontsize=20)
        title = 'Track number distribution (' + str(len(classes)) + '-class)'
        if n == 1: title += '\n(individually normalized)'
        weights = [len(n_tracks[n]) for n in classes] if n==1 else len(classes)*[len(labels)]
        weights = [len(n_tracks[n])*[100/weights[n]] for n in classes]
        plt.title(title, fontsize=20)
        label  =  ['class '+str(n)+' (mean: '+format(trk_mean[n],'3.1f')+')' for n in classes]
        plt.hist([n_tracks[n] for n in classes][::-1], bins=bins, lw=2, align='left',
                 weights=weights[::-1], label=label[::-1], histtype='step')
        plt.text(0.99, 0.05, '(sample: '+str(len(n_e))+' e)', {'color': 'black', 'fontsize': 12},
                 ha='right', va= 'center', transform=axes.transAxes)
        plt.legend(loc='upper right', fontsize=13)
    file_name = 'outputs/plots/tracks_number.png'; print('Printing:', file_name)
    plt.savefig(file_name)
    fig     = plt.figure(figsize=(22,6)); n = 1
    metrics = {'mean':(var_mean, 'Average'), 'max':(var_max, 'Maximum absolute'),
               'diff':(var_diff, 'Average difference')}
    #metrics = {'mean':(var_mean, 'Average'), 'max':(var_mean, 'Average'),
    #           'diff':(var_mean, 'Average')}
    for metric in metrics:
        plt.subplot(1, 3, n); axes = plt.gca(); n+=1
        n_e    = sum([len(metrics[metric][0][n]) for n in classes])
        x1, x2 = tracks_var[variable][metric+'_lim']
        bins   = np.arange(0.9*x1, 1.1*x2, (x2-x1)/100)
        plt.xlim([x1, x2])
        plt.title (metrics[metric][1] + ' value of ' + str(variable) + '\'s', fontsize=20)
        plt.xlabel(metrics[metric][1] + ' value'                            , fontsize=20)
        plt.ylabel('Normalized entries (%)'                                 , fontsize=20)
        #weights = [len(metrics[metric][0][n])*[100/len(metrics[metric][0][n])] for n in classes]
        weights = [len(metrics[metric][0][n])*[100/n_e] for n in classes]
        plt.hist([metrics[metric][0][n] for n in classes][::-1], weights=weights[::-1], stacked=False,
                 histtype='step', label=['class '+str(n) for n in classes][::-1], bins=bins, lw=2)
        plt.text(0.01, 0.97, '(sample: '+str(n_e)+' e)', {'color': 'black', 'fontsize': 12},
                 ha='left', va= 'center', transform=axes.transAxes)
        plt.legend(loc='upper right', fontsize=13)
    file_name = 'outputs/plots/tracks_'+str(variable)+'.png'; print('Printing:', file_name)
    plt.savefig(file_name)
