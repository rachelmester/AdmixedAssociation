import fire
import statsmodels.api as sm
from scipy.stats import bernoulli, binom, chi2
import numpy as np
import math
import tempfile
from os.path import join
import subprocess
import pandas as pd
from scipy import stats
import csv
from scipy.special import logit, expit
from scipy.optimize import fsolve

def create_genotypes(freq1, freq2, p_1, sim_dir = "../sim_data", n_indiv = 10000, sd_theta = 0.125):
    p_1 = 0.5 #expected value for global ancestry (probability of ancestry 1)    
    n_anc = 2 #this does not allow for any different number of ancestries at this time
    n_haplo = 2 #this does not allow for any different number of haplotypes at this time

    f = np.round([freq1, freq2], 2) #allele frequencies for allele 1 at each ancestry
    sd_theta = 0.125 #standard deviation for global ancestry for individuals

    #create fake data I can manipulate (this is for one SNP)   
    dir_name = f"{sim_dir}/{freq1}_{freq2}_{p_1}_{n_indiv}"
    subprocess.run(["mkdir", dir_name])

    theta = np.random.normal(p_1, sd_theta, n_indiv) #global ancestry array
    for i in range(0, n_indiv):
        if theta[i] <0:
            theta[i] = 0.0
        elif theta[i] > 1:
            theta[i] = 1.0
    lanc = np.zeros((n_indiv, n_haplo), dtype=np.int8) #local ancestry array (whether each allele is of ancestry 1)
    for i in range(0, n_indiv):
        lanc[i,:] = np.random.binomial(1, theta[i], 2)

    geno = np.ndarray((n_indiv, n_anc), dtype=np.int8) #genotype array for one SNP, rows for individuals, columns for haplotypes
    for i in range(0, n_indiv):
        for j in range(0, n_haplo):
            geno[i,j] = np.random.binomial(1, f[lanc[i,j]], 1) #how many of allele 1
    with open(f"{dir_name}/geno.csv.gz", 'w') as file:
        writer = csv.writer(file)
        writer.writerows(geno)
    with open(f"{dir_name}/lanc.csv.gz", 'w') as file:
        writer = csv.writer(file)
        writer.writerows(lanc)
    with open(f"{dir_name}/theta.csv.gz", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(theta)
        
def runAllSim(OR, n_pheno, geno_dirs, out_dir):
    summ = pd.DataFrame()
    for geno_dir in geno_dirs:
        geno = []
        with open(f"{geno_dir}/geno.csv.gz", 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                geno.append(list(np.array(row).astype(int)))
        geno = np.array(geno)
        lanc = []
        with open(f"{geno_dir}/lanc.csv.gz", 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                lanc.append(list(np.array(row).astype(int)))
        lanc = np.array(lanc)
        with open(f"{geno_dir}/theta.csv.gz", 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                theta = list(np.array(row).astype(float))
        theta = np.array(theta)
        score_df = pd.DataFrame()
        for sim in range(0, n_pheno):
            np.random.seed(sim)
            pheno = simulate_phenotype_cc_1snp(geno, lanc, [OR, OR], theta, 0.1)
            study_index = sample_case_control(pheno)
            study_pheno=pheno[study_index]
            study_lanc=lanc[study_index, :]
            study_geno=geno[study_index, :]
            study_theta=theta[study_index]
            path = "/u/project/pasaniuc/rmester/AdmixAssociationProject/tractor-response-main/software/bin/mixscore"
            scores = ["ADM", "ATT", "MIX", "SUM", "SNP1", "TRACTOR"]
            rls = pd.DataFrame()
            for score in scores:
                rls[score] = [run_test(geno, lanc, theta, pheno, score, mixscore_path=path)]
            score_df = pd.concat((score_df, rls))
        info = pd.DataFrame()
        for score in scores:
            if score == "ADM":
                thresh = 1e-5
            else:
                thresh = 5e-8
            power = len(score_df[score_df[score] < thresh]) / n_pheno
            info[score] = power
        summ = pd.concat((summ, info))
        info.to_csv(f"{out_dir}.csv.gz", index=False)
    return

def run_test(geno, lanc, theta, phenos, score, mixscore_path="/u/project/pasaniuc/rmester/AdmixAssociationProject/tractor-response-main/software/bin/mixscore"):
    n_snp = 1
    ploidy = 2, 
    n_sample = len(theta)
    
    if score != "TRACTOR":
        mix_pheno = np.reshape(phenos, (len(phenos),))
        mix_theta = np.reshape(theta, (len(theta),))
        mix_lanc = np.reshape((lanc[:,0] + lanc[:,1]), (1,len(lanc)))
        mix_geno = np.reshape((geno[:,0] + geno[:,1]), (1,len(geno)))
        tmp = tempfile.TemporaryDirectory()
        tmp_dir = tmp.name
        param_dir = tmp_dir

        #write genotype file
        with open("/".join((param_dir, "geno")), 'w') as f:
            for k in range(mix_geno.shape[0]):
                sample = mix_geno[k,:]
                f.writelines("".join([str(i) for i in sample]) + '\n')

        #write local ancestry file
        with open("/".join((param_dir, "anc")), 'w') as f:
            for k in range(mix_lanc.shape[0]):
                sample = mix_lanc[k,:]
                f.writelines("".join([str(i) for i in sample]) + '\n')

        #write phenotype file
        with open("/".join((param_dir, "pheno")), 'w') as f:
            f.writelines("".join([str(i) for i in mix_pheno]) + '\n')

        #write global ancestry file
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

        dof = 1
        if score == "SUM":
            dof = 2
        pval = chi2.sf(float(rls[0]), dof)
        
    else:
        pval = cc_tractor(phenos, lanc, geno, theta)
    return pval

        
def simulate_adm(beta_ratio, freq1, freq2, p_1, seed, n_pheno, sim_dir, n_indiv, h2 = 0.005, abs_beta = 1.0, thresh = 1e-5):
    np.random.seed(chunk)
    beta = [beta_ratio * abs_beta, abs_beta]
    
    f = np.round([freq1, freq2], 2) #allele frequencies for allele 1 at each ancestry
    
    #load data
    geno = []
    with open(f"sim_data/{f[0]}_{f[1]}_{p_1}_{n_indiv}/geno.csv.gz", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            geno.append(list(np.array(row).astype(int)))
    geno = np.array(geno)
    lanc = []
    with open(f"sim_data/{f[0]}_{f[1]}_{p_1}_{n_indiv}/lanc.csv.gz", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            lanc.append(list(np.array(row).astype(int)))
    lanc = np.array(lanc)
    with open(f"sim_data/{f[0]}_{f[1]}_{p_1}_{n_indiv}/theta.csv.gz", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            theta = list(np.array(row).astype(float))
    theta = np.array(theta)

    #simulate the phenos
    score_df = pd.DataFrame()
    for k in range(0, n_pheno):
        phenos = simulate_phenotype_quant_1snp(geno, lanc, beta, theta, h2)

        #reshape parameters
        standard_phenos = (phenos - np.mean(phenos))/ np.std(phenos)    
        tractor_geno = convert_anc_count(geno, lanc)
        shape_lanc = np.reshape(lanc[:,0] + lanc[:,1], (n_indiv, 1)).astype(int)

        score_df = pd.concat([score_df, adm(pheno = standard_phenos, anc = shape_lanc, geno = tractor_geno, theta = theta)])
    ADM_power = len(score_df[score_df["ADM"] < thresh]) / n_pheno
    info = pd.DataFrame({"GA" :[p_1], "f0": [f[0]], "f1": [f[1]], "b0": [beta[0]], "b1": [beta[1]], "ADM": [ADM_power]})
        
    rls_dir = f"{sim_dir}/{f[0]}_{f[1]}_{p_1}_{n_indiv}/{beta[0]}_{beta[1]}"
    if abs_beta != 1.0:
        rls_dir = rls_dir + f"_ab_{abs_beta}"
    if h2 != 0.005:
        rls_dir = rls_dir + f"_h2_{h2}"
    info.to_csv(f"{rls_dir}_ADM_rls.csv.gz", index=False)
    

def simulate_specific_ind(beta_ratio, freq1, freq2, p_1, seed, n_pheno, sim_dir, n_indiv, h2 = 0.005, abs_beta = 1.0, thresh = 5e-8):
    np.random.seed(chunk)
    beta = [beta_ratio * abs_beta, abs_beta]
    
    f = np.round([freq1, freq2], 2) #allele frequencies for allele 1 at each ancestry
    
    #load data
    geno = []
    with open(f"sim_data/{f[0]}_{f[1]}_{p_1}_{n_indiv}/geno.csv.gz", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            geno.append(list(np.array(row).astype(int)))
    geno = np.array(geno)
    lanc = []
    with open(f"sim_data/{f[0]}_{f[1]}_{p_1}_{n_indiv}/lanc.csv.gz", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            lanc.append(list(np.array(row).astype(int)))
    lanc = np.array(lanc)
    with open(f"sim_data/{f[0]}_{f[1]}_{p_1}_{n_indiv}/theta.csv.gz", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            theta = list(np.array(row).astype(float))
    theta = np.array(theta)

    #simulate the phenos
    score_df = pd.DataFrame()
    for k in range(0, n_pheno):
        phenos = simulate_phenotype_quant_1snp(geno, lanc, beta, theta, h2)

        #reshape parameters
        standard_phenos = (phenos - np.mean(phenos))/ np.std(phenos)    
        tractor_geno = convert_anc_count(geno, lanc)
        shape_lanc = np.reshape(lanc[:,0] + lanc[:,1], (n_indiv, 1)).astype(int)

        score_df = pd.concat([score_df, tractor(pheno = standard_phenos, anc = shape_lanc, geno = tractor_geno, theta = theta)])

    ATT_power = len(score_df[score_df["ATT"] < thresh]) / n_pheno
    TRACTOR_power = len(score_df[score_df["TRACTOR"] < thresh]) / n_pheno
    info = pd.DataFrame({"GA" :[p_1], "f0": [f[0]], "f1": [f[1]], "b0": [beta[0]], "b1": [beta[1]], "ATT": [ATT_power], "TRACTOR": [TRACTOR_power]})
        
    rls_dir = f"{sim_dir}/{f[0]}_{f[1]}_{p_1}_{n_indiv}/{beta[0]}_{beta[1]}"
    if abs_beta != 1.0:
        rls_dir = rls_dir + f"_ab_{abs_beta}"
    if h2 != 0.005:
        rls_dir = rls_dir + f"_h2_{h2}"
    info.to_csv(f"{rls_dir}_rls.csv.gz", index=False)

def simulate_phenotype_quant_1snp(geno, lanc, beta, theta, h2):
    n_indiv = geno.shape[0]
    assert len(theta) == n_indiv
    beta=np.array(beta,ndmin=2).reshape((-1,1))
    snp_geno = convert_anc_count(geno, lanc)
    # allelic risk effect size x number of minor alleles
    snp_phe_g = np.dot(snp_geno, beta)
    #add the environmental component
    snp_phe = np.zeros_like(snp_phe_g, dtype=np.int8)
    var_g = np.std(snp_phe_g)**2
    if var_g == 0:
        var_e = 1.0
    else:
        var_e = var_g*(1-h2)/h2 
    snp_phe = np.add(snp_phe_g, np.random.normal(scale = math.sqrt(var_e), size = (n_indiv,1)))
    return snp_phe

def simulate_phenotype_cc_1snp(geno, lanc, beta, theta, case_prevalence):
    n_indiv = geno.shape[0]
    assert len(theta) == n_indiv
    beta=np.array(beta,ndmin=2).reshape((-1,1))
    snp_geno = convert_anc_count(geno, lanc)
    # allelic risk effect size x number of minor alleles
    snp_phe_g = np.dot(snp_geno, np.log(beta))
    snp_phe = np.zeros_like(snp_phe_g, dtype=np.int8)
    # find an intercept, such that the expectation is case_prevalence.
    func = lambda b: np.mean(expit(b + snp_phe_g)) - case_prevalence
    intercept = fsolve(func, logit(case_prevalence))
    snp_phe = np.random.binomial(1, expit(intercept + snp_phe_g))
    return snp_phe

def convert_anc_count(phgeno, anc):
    n_indiv = anc.shape[0]
    n_snp = anc.shape[1] // 2
    phgeno = phgeno.reshape((n_indiv * 2, n_snp))
    anc = anc.reshape((n_indiv * 2, n_snp))

    geno = np.zeros((n_indiv, n_snp * 2), dtype=np.int8)
    for indiv_i in range(n_indiv):
        for haplo_i in range(2 * indiv_i, 2 * indiv_i + 2):
            for anc_i in range(2):
                anc_snp_index = np.where(anc[haplo_i, :] == anc_i)[0]
                geno[indiv_i, anc_snp_index + anc_i * n_snp] += phgeno[
                    haplo_i, anc_snp_index
                ]
    return geno

def sample_case_control(pheno, control_ratio=1):
    case_index = np.where(pheno == 1)[0]
    control_index = np.random.choice(
        np.where(pheno == 0)[0],
        size=int(len(case_index) * control_ratio),
        replace=False,
    )
    study_index = np.sort(np.concatenate([case_index, control_index]))
    return study_index

def tractor(pheno, anc, geno, theta):
    # local ancestry
    m1_design = np.hstack([sm.add_constant(anc), theta[:, np.newaxis]])
    m1_model = sm.GLS(pheno, m1_design).fit(disp=0, maxiter=200)

    # local ancestry + genotype (regardless of ancestry)
    m2_design = np.hstack([m1_design, geno.mean(axis=1)[:, np.newaxis]])
    m2_model = sm.GLS(pheno, m2_design).fit(disp=0, maxiter=200)

    # local ancestry + genotype (ancestry aware)
    m3_design = np.hstack([m1_design, geno])
    m3_model = sm.GLS(pheno, m3_design).fit(disp=0, maxiter=200)

    # genotype (regardless of ancestry)
    att_design = np.hstack(
        [sm.add_constant(geno.mean(axis=1)[:, np.newaxis]), theta[:, np.newaxis]]
    )
    att_model = sm.GLS(pheno, att_design).fit(disp=0, maxiter=200)
    
    m0_design = np.hstack([theta[:, np.newaxis]])
    m0_model = sm.GLS(pheno, m0_design).fit(disp=0, maxiter=200)
                           
    rls = pd.DataFrame({"ATT": [att_model.pvalues[1]], "TRACTOR": [stats.chi2.sf(-2 * (m1_model.llf - m3_model.llf), 2)]})
    return rls

def adm(pheno, anc, geno, theta):
    m1_design = np.hstack([sm.add_constant(anc), theta[:, np.newaxis]])
    m1_model = sm.GLS(pheno, m1_design).fit(disp=0, maxiter=200)
    rls = pd.DataFrame({"ADM": [m1_model.pvalues[1]]})
    return rls

def cc_tractor(pheno, anc, geno, theta):
    # local ancestry
    m1_design = np.hstack([sm.add_constant(anc), theta[:, np.newaxis]])
    m1_model = sm.Logit(pheno, m1_design).fit(disp=0, method="bfgs", maxiter=200)

    # local ancestry + genotype (ancestry aware)
    m3_design = np.hstack([m1_design, geno])
    m3_model = sm.Logit(pheno, m3_design).fit(
        disp=0,
        method="bfgs",
        maxiter=200,
        start_params=np.concatenate([m1_model.params, [0.0, 0.0]]),
    )

    return chi2.sf(-2 * (m1_model.llf - m3_model.llf), 2)

    
if __name__ == "__main__":
    fire.Fire()