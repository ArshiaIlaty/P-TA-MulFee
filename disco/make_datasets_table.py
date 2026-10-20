"""
Datasets description table for the paper, with clean names (no surv_/tab_
provenance prefixes) and source citations. Writes to tables/comprehensive/paper/:
  datasets_table.tex  -- Dataset | Domain | #inst | #feat | #cls | Source\\cite
  datasets.bib        -- source references (collections + key originals)
  datasets_preview.pdf

Citations are at the collection level we actually sourced each dataset from
(UCI, SurvSet, TabArena/OpenML) plus well-known originals; verify keys against
the paper's .bib before submission.
"""
import os, sys, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_raw")
OUT = os.path.join(HERE, "tables", "comprehensive", "paper")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
from data import infer_types, TARGET

DS_ORDER = ["heart_failure", "breast_cancer", "liver_disorders", "parkinsons",
            "obesity", "diabetes", "german_credit", "bank_marketing", "census_income",
            "surv_melanoma", "surv_pbc", "surv_pbc3", "surv_retinopathy",
            "surv_whas500", "surv_gbsg2", "surv_colon", "surv_grace", "surv_trace",
            "surv_framingham", "tab_blood_transfusion",
            "tab_qsar_biodeg", "tab_maternal_risk", "tab_website_phishing"]
DISPLAY = {
    "heart_failure": "Heart Failure", "breast_cancer": "Breast Cancer",
    "liver_disorders": "Liver Disorders", "parkinsons": "Parkinsons",
    "obesity": "Obesity", "diabetes": "Diabetes", "german_credit": "German Credit",
    "bank_marketing": "Bank Marketing", "census_income": "Census Income",
    "surv_melanoma": "Melanoma", "surv_pbc": "PBC", "surv_pbc3": "PBC3",
    "surv_retinopathy": "Retinopathy", "surv_whas500": "WHAS500",
    "surv_gbsg2": "GBSG2", "surv_colon": "Colon", "surv_grace": "GRACE",
    "surv_trace": "TRACE", "surv_framingham": "Framingham",
    "tab_blood_transfusion": "Blood Transfusion",
    "tab_qsar_biodeg": "QSAR Biodeg", "tab_maternal_risk": "Maternal Risk",
    "tab_website_phishing": "Website Phishing",
}
DOMAIN = {d: "Clinical" for d in DS_ORDER}
DOMAIN.update({"german_credit": "Finance", "bank_marketing": "Finance",
               "census_income": "Finance",
               "tab_qsar_biodeg": "Chemistry", "tab_website_phishing": "Security"})
# source citation key per dataset (collection we sourced it from / original)
CITE = {
    "heart_failure": "chicco2020ml", "breast_cancer": "uci", "liver_disorders": "uci",
    "parkinsons": "little2007exploiting", "obesity": "palechor2019dataset",
    "diabetes": "kaggle_diabetes", "german_credit": "uci", "bank_marketing": "moro2014data",
    "census_income": "kohavi1996scaling",
}
CITE.update({d: "drysdale2022survset" for d in DS_ORDER if d.startswith("surv_")})
CITE.update({d: "erickson2025tabarena" for d in DS_ORDER if d.startswith("tab_")})

BIB = r"""@misc{uci, title={{UCI} Machine Learning Repository}, author={Dua, Dheeru and Graff, Casey}, year={2019}, howpublished={\url{https://archive.ics.uci.edu/ml}}}
@article{chicco2020ml, title={Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone}, author={Chicco, Davide and Jurman, Giuseppe}, journal={BMC Medical Informatics and Decision Making}, volume={20}, number={1}, pages={16}, year={2020}}
@article{little2007exploiting, title={Exploiting nonlinear recurrence and fractal scaling properties for voice disorder detection}, author={Little, Max A and McSharry, Patrick E and Roberts, Stephen J and Costello, Declan A and Moroz, Irene M}, journal={BioMedical Engineering OnLine}, volume={6}, number={1}, pages={23}, year={2007}}
@article{palechor2019dataset, title={Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico}, author={Palechor, Fabio Mendoza and de la Hoz Manotas, Alexis}, journal={Data in Brief}, volume={25}, pages={104344}, year={2019}}
@misc{kaggle_diabetes, title={Diabetes prediction dataset}, author={Mustafa, Mohammed}, year={2023}, howpublished={\url{https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset}}}
@article{moro2014data, title={A data-driven approach to predict the success of bank telemarketing}, author={Moro, S{\'e}rgio and Cortez, Paulo and Rita, Paulo}, journal={Decision Support Systems}, volume={62}, pages={22--31}, year={2014}}
@inproceedings{kohavi1996scaling, title={Scaling up the accuracy of naive-{B}ayes classifiers: a decision-tree hybrid}, author={Kohavi, Ron}, booktitle={KDD}, pages={202--207}, year={1996}}
@article{drysdale2022survset, title={{SurvSet}: An open-source time-to-event dataset repository}, author={Drysdale, Erik}, journal={arXiv preprint arXiv:2203.03094}, year={2022}}
@article{erickson2025tabarena, title={{TabArena}: A living benchmark for machine learning on tabular data}, author={Erickson, Nick and others}, journal={arXiv preprint}, year={2025}}
"""


def main():
    rows = []
    for ds in DS_ORDER:
        df = pd.read_csv(os.path.join(RAW, f"{ds}.csv"))
        df[TARGET] = df[TARGET].astype(str)
        num, cat = infer_types(df, TARGET)
        rows.append((DISPLAY[ds], DOMAIN[ds], len(df), len(num) + len(cat),
                     df[TARGET].nunique(), CITE[ds]))
    body = []
    body.append(r"\begin{table}[t]\centering\small")
    body.append(r"\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.05}")
    body.append(r"\caption{Benchmark datasets (23 total: 9 from the original study plus "
                r"10 clinical time-to-event cohorts from SurvSet and 4 from TabArena). "
                r"Time-to-event cohorts use the event indicator as a binary target.}")
    body.append(r"\label{tab:datasets}")
    body.append(r"\begin{tabular}{@{}llrrrl@{}}\toprule")
    body.append(r"\textbf{Dataset} & \textbf{Domain} & \textbf{\#Inst.} & \textbf{\#Feat.} & \textbf{\#Cls.} & \textbf{Source} \\ \midrule")
    for name, dom, n, f, c, cite in rows:
        body.append(f"{name} & {dom} & {n} & {f} & {c} & \\cite{{{cite}}} \\\\")
    body.append(r"\bottomrule\end{tabular}\end{table}")
    tex = "\n".join(body)
    open(os.path.join(OUT, "datasets_table.tex"), "w").write(tex + "\n")
    open(os.path.join(OUT, "datasets.bib"), "w").write(BIB)
    doc = (r"\documentclass[10pt]{article}"
           r"\usepackage[margin=1.5cm]{geometry}\usepackage{booktabs,url}"
           r"\usepackage[numbers]{natbib}" "\n\\begin{document}\n" + tex +
           "\n\\bibliographystyle{plainnat}\n\\bibliography{datasets}\n\\end{document}\n")
    open(os.path.join(OUT, "datasets_preview.tex"), "w").write(doc)
    for cmd in (["pdflatex"], ["bibtex"], ["pdflatex"], ["pdflatex"]):
        subprocess.run(cmd + (["-interaction=nonstopmode", "datasets_preview"]
                              if cmd[0] == "pdflatex" else ["datasets_preview"]),
                       cwd=OUT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote datasets_table.tex, datasets.bib, datasets_preview.pdf")
    for name, dom, n, f, c, cite in rows:
        print(f"  {name:20} {dom:10} n={n:<7} feat={f:<3} cls={c} [{cite}]")


if __name__ == "__main__":
    main()
