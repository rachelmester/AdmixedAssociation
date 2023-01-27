import pandas as pd
import csv
import numpy as np
import admix #see for instructions on installing this package at https://github.com/KangchengHou/admix-kit
import fire
import math
import tempfile
import subprocess
from scipy.stats import chi2

def empirical_rhet(phenos, results_dir, out_dir, thresh = 5e-8):
    ratios_dict = {}
    ratios_array = np.array([])
    tractor_sig = np.array([])
    att_sig = np.array([])
    for pheno in phenos:
        ratio_sig = np.array([])
        for chrom in range(1, 23):
            rls = pd.read_csv(f"{results_dir}/{pheno}/{chrom}_summary.csv.gz")
            att_p = rls["ATT_P"]
            tractor_p = rls["TRACTOR_P"]
            att_score = rls["ATT_score"]
            tractor_score = rls["TRACTOR_score"]
            ratio = rls["b1/b2"]

            for i in range(0, len(att_p)):
                if att_p[i] <= thresh or tractor_p[i] <= thresh:
                    ratio_sig = np.append(ratio_sig, ratio[i])
                    tractor_sig = np.append(tractor_sig, tractor_score[i])
                    att_sig = np.append(att_sig, att_score[i])

        ratios_dict[pheno] = ratio_sig
        ratios_array = np.concatenate((ratios_array, ratio_sig))       

    ratios_array = ratios_array[np.isfinite(ratios_array)]
    ratios = pd.DataFrame()
    phenolist = np.array([])
    ratiolist = np.array([])
    for pheno in phenos:
        phenolist = np.concatenate((phenolist, np.repeat([pheno], len(ratios_dict[pheno]))))
        ratiolist = np.concatenate((ratiolist, ratios_dict[pheno]))
    ratios["pheno"] = phenolist
    ratios["ratio"] = ratiolist
    ratios.to_csv(f"{out_dir}/ratiolist.csv.gz", index = False)
    
    sigs = pd.DataFrame()
    sigs["att"] = att_sig
    sigs["tractor"] = tractor_sig
    sigs.to_csv(f"{out_dir}/scorelist.csv.gz", index = False)

def track_MAFs(geno_dir, out_dir):
    diff_array = np.array([])
    n_snp_ukbb = 0
    numerator = 0
    for chrom in range(1, 23):
        impute_file = f"{geno_dir}/chr{chrom}"
        impute_dset = admix.io.read_dataset(pfile = impute_file)
        n_snp_ukbb = n_snp_ukbb + dset.n_snp
        snp_info  = impute_dset.snp
        diff = snp_info["LANC_FREQ1"] - snp_info["LANC_FREQ2"]
        diff_array = np.append(diff_array, diff)
        numerator = numerator + np.sum(impute_dset.lanc).compute()
        if chrom == 1:
            n_indiv = len(impute_dset.indiv)
    diffs = pd.DataFrame(diff_array)
    diffs.to_csv(f"{out_dir}/info_realdiffs.csv.gz", index = False)

    info = pd.DataFrame()
    info["n_snp"] = [n_snp_ukbb]
    info["n_indiv"] = [n_indiv]
    info["p_1"] = [numerator / (2 * n_snp_ukbb * n_indiv)]
    info.to_csv(f"{out_dir}/info_realinfo.csv.gz", index = False)
    
def compile_manhattan(results_dir, thresh = 1e-5):
    #compile real data for manhattan plots
    for pheno in phenos:
        tractor = pd.DataFrame()
        att = pd.DataFrame()
        tractor_p = []
        tractor_snp = []
        att_snp = []
        att_p = []
        thresh = 1e-5
        for chrom in range(1, 23):
            summ = pd.read_csv(f"{results_dir}/{pheno}/{chrom}_summary.csv.gz")
            snps = summ["SNP_I"]
            tractor2 = summ["TRACTOR_P"]
            att2 = summ["ATT_P"]
            for i in range(0, len(tractor2)):
                if tractor2[i] <= thresh:
                    tractor_p.append(tractor2[i])
                    tractor_snp.append(snps[i])
                if att2[i] <= thresh:
                    att_p.append(att2[i])
                    att_snp.append(snps[i])
        tractor["p"] = tractor_p
        att["p"] = att_p
        tractor["SNP"] = tractor_snp
        att["SNP"] = att_snp

        x = att["SNP"]
        y = list(x)
        z = np.copy(y)
        a = np.copy(y)
        for i in range(0, len(x)):
            b = y[i].split(":")
            z[i] = b[0].split("r")[1]
            a[i] = b[1]
        att["chrom"] = z
        att["pos"] = a
        att["-log10p"] = -np.log10(att["p"])
        att["score"] = np.repeat("ATT", len(att))

        x = tractor["SNP"]
        y = list(x)
        z = np.copy(y)
        a = np.copy(y)
        for i in range(0, len(x)):
            b = y[i].split(":")
            z[i] = b[0].split("r")[1]
            a[i] = b[1]
        tractor["chrom"] = z
        tractor["pos"] = a
        tractor["-log10p"] = -np.log10(tractor["p"])
        tractor["score"] = np.repeat("TRACTOR", len(tractor))

        tractor.to_csv(f'{results_dir}/{pheno}/tractor_all.csv.gz', index=False)
        att.to_csv(f'{results_dir}/{pheno}/att_all.csv.gz', index=False)
        
def find_independent(phenos, pthresh, kthresh, results_dir):  
    for pheno in phenos:
        for score in ["TRACTOR", "ATT"]:
            sig = pd.DataFrame()
            for chrom in range(1, 23):
                summ = pd.read_csv(f'{results_dir}/{pheno}/{chrom}_summary.csv.gz')
                sumstats = summ[[f"{score}_P", "SNP_I", "TRACTOR_G1_BETA", "TRACTOR_G2_BETA"]].copy()
                sumstats["chrom"] = [int(s[3:].split(':')[0]) for s in sumstats.SNP_I]
                sumstats["pos"] = [int(s[3:].split(':')[1]) for s in sumstats.SNP_I]
                sumstats = sumstats.sort_values(f'{score}_P').reset_index()
                pos = np.array([])
                i = 0
                while sumstats[f"{score}_P"][i] <= p_thresh:
                    indep = True
                    for k in pos:
                        if sumstats["pos"][i] - k < k_thresh:
                            indep = False
                    if indep:
                        pos = np.append(pos, sumstats["pos"][i])
                    i = i + 1
                sig = pd.concat((sig, sumstats[sumstats["pos"].isin(pos)]))
            sig.to_csv(f"{results_dir}/{pheno}/{score}f.csv.gz")
            
def run_gwas_wrapper(geno_file, pheno_file, out_file, chunk_size):
    ph = pd.read_csv(pheno_file, sep='\t')
    dset = admix.io.read_dataset(pfile = geno_file)
    
    rls = run_gwas(dset, ph, "linear", chunk_size)
    rls.to_csv(out_file, index=False)
    
def run_gwas(dset_all: admix.dataset, ph: pd.DataFrame, family: str, chunk_size: int):
    
    #filter dataset for phenotyped individuals only
    ph.index = ph['INDIV']
    inclusion = []
    for i in range(len(dset_all.indiv.index)):
        if dset_all.indiv.index[i] in ph.index:
            inclusion.append(i)
    dset = dset_all[:, inclusion]
    dset.append_indiv_info(ph, True)
    
    #normalize covariates and phenotype
    covar_cols = ph.columns[2:]
    covar = dset.indiv[covar_cols].values
    for i in range(0, np.shape(covar)[1]):
        col = covar[:,i]
        covar[:,i] = (col - np.nanmean(col)) / np.nanstd(col)
    dset.indiv[covar_cols] = covar
    if family == "linear":
        dset.indiv["PHENO"] = quantile_normalize(dset.indiv["PHENO"])
    
    #filter dataset for SNPs within the chunk window
    n_chunk = math.ceil(dset.n_snp / chunk_size)
    summ_rls = pd.DataFrame()

    for chunk in range(1,n_chunk+1):
        if (chunk * chunk_size) > dset.n_snp:
            dset_small = dset[((chunk-1)*chunk_size):dset.n_snp, :]
        else:
            dset_small = dset[((chunk-1)*chunk_size):(chunk*chunk_size), :]
        rls = pd.DataFrame()
        for method in ["SNP1", "ADM", "ATT", "TRACTOR"]:
            dict_rls = admix.assoc.marginal(
                dset=dset_small,
                pheno=dset_small.indiv["PHENO"],
                family=family,
                method=method,
                cov=dset_small.indiv[covar_cols].values
            )
            chisq = []
            for pval in dict_rls["P"]:
                chisq.append(chi2.isf(pval, 1))
            rls[f"{method}_score"] = chisq
            dict_rls["score"] = chisq
            for label, content in dict_rls.items():
                rls[f"{method}_{label}"] = np.array(content)
        rls["SUM_score"] = np.add(rls["SNP1_score"], rls["ADM_score"])
        pval = []
        chisq = []
        for score in rls["SUM_score"]:
            p = chi2.sf(score, 2)
            pval.append(p)
            chisq.append(chi2.isf(p, 1))
        rls["SUM_score"] = chisq
        rls["SUM_P"] = pval
        rls["SNP_I"] = dict_rls.index
        summ_rls = pd.concat((summ_rls, rls))
    
    quotients = []
    b1 = summ_rls["TRACTOR_G1_BETA"]
    b2 = summ_rls["TRACTOR_G2_BETA"]
    for i in range(0, len(b1)):
        if b2[i] == 0:
            quotients.append(0)
        else:
            quotients.append(b1[i]/b2[i])
    summ_rls["b1/b2"] = quotients
    return summ_rls

def quantile_normalize(val):
    from scipy.stats import rankdata, norm

    val = np.array(val)
    non_nan_index = ~np.isnan(val)
    results = np.full(val.shape, np.nan)
    results[non_nan_index] = norm.ppf(
        (rankdata(val[non_nan_index]) - 0.5) / len(val[non_nan_index])
    )
    return results
    
def mixscore(dset, phenos, score, mixscore_path="/u/project/pasaniuc/rmester/AdmixAssociationProject/tractor-response-main/software/bin/mixscore"):
    
    n_snp, n_sample, ploidy = dset.geno.shape
    tmp = tempfile.TemporaryDirectory()
    tmp_dir = tmp.name
    param_dir = tmp_dir
    
    #write genotype file
    mix_geno = (dset.geno[:,:,0]+dset.geno[:,:,1])
    with open("/".join((param_dir, "geno")), 'w') as f:
        for k in range(mix_geno.shape[0]):
            sample = mix_geno[k,:].compute()
            f.writelines("".join([str(i) for i in sample]) + '\n')
        
    #write local ancestry file
    mix_lanc = (dset.lanc[:,:,0] + dset.lanc[:,:,1])
    with open("/".join((param_dir, "anc")), 'w') as f:
        for k in range(mix_lanc.shape[0]):
            sample = mix_lanc[k,:].compute()
            f.writelines("".join([str(i) for i in sample]) + '\n')
        
    #write phenotype file
    if score in ["ATT", "ADM", "SNP1", "SUM", "MIX"]:
        with open("/".join((param_dir, "pheno")), 'w') as f:
            f.writelines("".join([str(i) for i in phenos]) + '\n')
    else:
        with open("/".join((param_dir, "pheno")), 'w') as f:
            f.writelines(" ".join([str(i) for i in phenos]) + '\n')

    #write global ancestry file
    mix_theta = (mix_lanc.sum(axis = 0) / (n_snp * ploidy)).compute()
    with open("/".join((param_dir, "theta")), 'w') as f:
        f.writelines([str(mix_theta[k]) + '\n' for k in range(mix_theta.shape[0])])
            
    #write parameters file   
    param = {"nsamples": str(n_sample),
               "nsnps": str(n_snp),
               "phenofile": "/".join((param_dir,"pheno")),
               "ancfile": "/".join((param_dir,"anc")),
               "genofile": "/".join((param_dir, "geno")),
               "thetafile": "/".join((param_dir, "theta")),
               "outfile": "/".join((param_dir, "out"))}
    with open("/".join((param_dir, "param")), 'w') as f:
        f.writelines([k + ':' + param[k] + '\n' for k in param])
        
    #run software
    cmd = ' '.join([mixscore_path, score, param_dir+"/param"])
    subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    with open(param["outfile"]) as f:
        out = [line.strip() for line in f.readlines()]
        rls = out
    tmp.cleanup()
    return rls


if __name__ == "__main__":
    fire.Fire()