def get_kl_div(p,q):
    div_arr = np.where(np.logical_and(p>0,q>0),kl_div(p,q),0)
    return np.sum(div_arr)


def plot_kl_div_toy(x1, x2, label1, label2, w1=None, w2=None, name="feature", title = "", bins=50, outdir="plots", *args, **kwargs):
    
    colors = ['blue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    
    fig, ax1 = plt.subplots(figsize=(8,6))
    c0, cbins, _ = ax1.hist(x1, bins = bins, density = True, weights=w1, histtype='step', color=colors[2], label=label1)
    c1, _, _ = ax1.hist(x2, bins = cbins, density = True, weights=w2, histtype='stepfilled', alpha = 0.3, color=colors[2], label=label2)
    kl_div = get_kl_div(c0,c1)
    ax1.hist(x2, bins = bins, density = True, histtype='stepfilled', alpha = 0, color=colors[2], label=f"kl div={kl_div:.3f}")
    ax1.set_title(title, fontsize = 14)
    ax1.set_xlabel(name)
    plt.legend(fontsize = 10)
    plt.show
    plot_name = f"{outdir}/{label1}_{label2}_{name}.png"
    plt.savefig(plot_name.replace(" ", "_"))
    plt.close()
        
        

def plot_kl_div_data_reweight(data_train, data_true, data_gen, weights, data_gen_from_truth=None, MC_true=None, name="data_reweight", title="", ymin=-6, ymax=10, outdir="./", *args, **kwargs):
    
    # data_train is the data in CR
    # data_true is the data in SR
    # data_gen is the predicted data in SR
    # weights is used to reweight MC to data
    # 
    # optional:
    # MC_true
    # data_gen_from_truth
    
    
    colors = ['blue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    
    bins = np.linspace(ymin, ymax, 50)
    fig, ax1 = plt.subplots(figsize=(10,6))

    ax1.hist(data_train, bins = bins, density = True, histtype='step', ls="--", color=colors[0], label=f"data in CR")
    
    c0, cbins, _ = ax1.hist(data_true, bins = bins, density = True, histtype='step', color=colors[0], label=f"data in SR")
    
    if MC_true is not None:
        ax1.hist(MC_true, bins = bins, density = True, histtype='step', color=colors[1], label=f"MC in SR")
    
    ax1.hist(data_gen, bins = bins, density = True, histtype='stepfilled', alpha = 0.3, color=colors[1], label=f"no weight pred bkg in SR from MC")
    
    c1, cbins, _ = ax1.hist(data_gen, bins = bins, density = True, weights=weights, histtype='stepfilled', alpha = 0, color=colors[0])
    kl_div = get_kl_div(c0,c1)
    ax1.hist(data_gen, bins = bins, density = True, weights=weights, histtype='stepfilled', alpha = 0.3, color=colors[0], label=f"pred bkg in SR from MC (kl div from data = {kl_div:.3f})")
    
    if data_gen_from_truth is not None:
        c2, cbins, _ = ax1.hist(data_gen_from_truth, bins = bins, density = True, histtype='stepfilled', alpha = 0, color=colors[3])
        kl_div = get_kl_div(c0,c2)
        ax1.hist(data_gen_from_truth, bins = bins, density = True, histtype='stepfilled', alpha = 0.3, color=colors[3], label=f"pred bkg in SR from data (kl div from data = {kl_div:.3f})")
    
    ax1.set_title(f"True vs predicted background in SR {title}", fontsize = 14)
    ax1.set_xlabel("x")
    plt.legend(loc='upper left', fontsize = 9)
    plt.show
    plot_name = f"{outdir}/{name}.png"
    plt.savefig(plot_name.replace(" ", "_"))
    plt.close()
  

def plot_kl_div_phys(x1, x2, label1, label2, w1=None, w2=None, name="feature", tag = "", bins=50, outdir="plots", *args, **kwargs):
    
    colors = ['blue', 'slategrey', 'teal', 'limegreen', 'olivedrab', 'gold', 'orange', 'salmon']
    
    fig, ax = plt.subplots(2, figsize=(8,8), gridspec_kw={'height_ratios': [3, 1]})

    c0, cbins, _ = ax[0].hist(x1, bins = bins, density = True, weights=w1, histtype='step', lw=2, color=colors[2], label=label1)
    c1, cbins, _ = ax[0].hist(x2, bins = bins, density = True, weights=w2, histtype='stepfilled', alpha = 0.2, color=colors[1], label=label2)

    # ratio of gen SR vs true SR
    gen_bkg_SR = np.array(c1)
    target_bkg_SR = np.array(c0)
    r_bkg = np.divide(gen_bkg_SR, target_bkg_SR, out=np.full_like(gen_bkg_SR, np.nan), where=(target_bkg_SR != 0))

    # plot ratio
    ax[1].plot(cbins[:-1], r_bkg, color='slategrey', marker='.', lw=2)
    ax[1].axhline(y=1, color='black', linestyle='-')
    ax[1].set_xlabel(name, fontsize=14)
    ax[1].set_ylabel("Ratio to truth", fontsize=16)
    ax[1].set_ylim(0.5, 1.5)
    ax[1].tick_params(axis='both', which='major', labelsize=12)

    ax[0].set_ylabel("Events (a.u.)", fontsize=14)
    ax[0].set_xticks([]) 
    ax[0].legend(fontsize=16)
    
    plot_name = f"{outdir}/{label1}_{label2}_{tag}.png"
    plot_name = plot_name.replace(" ", "_")
    plt.savefig(plot_name)
    print(f"MAF plots saved as {plot_name}")
    plt.close()