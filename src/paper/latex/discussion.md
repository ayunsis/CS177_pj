---
marp: true
theme: default
style: |
  /* Custom Marp Theme */

  section {

  }

  /* Header style */
  h1, h2, h3, h4, h5, h6 {
    color: #1E90FF; /* blue headers */
  }

  /* Paragraph style */
  p {
    line-height: 1.6;
  }

  /* List style */
  ul, ol {
    margin-left: 20px;
  }

  /* Blockquote style */
  blockquote {
    border-left: 4px solid #FFD700; /* Yellow border */
    padding-left: 16px;
    color: #68228B; /* Blue text */
    font-style: italic;
  }

  /* Code block style */
  pre {
    background-color: #f0f8ff; /* Light blue background */
    color: #9400D3; /* Pale blue text */
    padding: 16px;
    border-radius: 4px;
    overflow: auto;
  }

  /* Inline code style */
  code {
    background-color: #FFFFE0; /* Light yellow background */
    color: #8470FF; /* Pale blue text */
    padding: 2px 4px;
    border-radius: 3px;
  }

  /* Footer style */
  footer {
    position: absolute;
    bottom: 20px;
    width: 100%;
    text-align: center;
    color: #FFD700; /* Yellow footer */
    font-size: 0.8em;
  }

  .align-right {
    float: right;
    margin-left: 20px;
    max-width: 50%; /* Adjust as needed */
  }
---
# Midterm Project Discussion
- Assessing the Reliability of AlphaFold3 Predictions for
Protein-Ligand Affinity Prediction via Sfcnn

> Delivered by Guo Yu, Yiming Wu, Yiyang Tan
---
# Overview
- Reproduce the result of Sfcnn on CASF-2016 coreset
- Generate corresponding result using AF3
- Evaluate AF3 structures' PLA (protein-ligand affinity) on Sfcnn

---
# Part 1: Reproduction
## 1.1: Dataset
- training set: PDBbind-2019 refined set, **4852** proteins, **266** excluded overlaps
- testing set: CASF-2016 core set, **285** proteins
- validation: *15%* training set, **752** proteins
- input: **.pdb** and **.mol2** 
- output: protein 3D **grid**

> implemented by: Guo Yu
---
# Part 1: Reproduction
## 1.2: Data Pipeline
### Featurization
- parse by **Openbabel.pybel**
- **One hot** encoding
- **14** atom types
- (4852, 20, 20, 20, 28) 
![bg right:54% left: 80% fit](./discussion_img/one_hot.png)
> implemented by: Guo Yu

---
# Part 1: Reproduction
## 1.2: Data Pipeline
### Augmentation
- 9 random rotations + 1 original complex
- (48520, 20, 20, 20, 28)

### Storage
- Unfeasible using **.pkl** (numpy.concatenate)
- **.h5** (HDF5) for large storage

> implemented by: Guo Yu

---
# Part 1: Reproduction
## 1.3: NetWork 
- input: 3D grid, output: PLA score
- 3D convolution Neural Net, implemented by **pytorch**
- FC final **weight decay** <-> **L2 regularization** 
![bg right:54% left: 60% fit](./discussion_img/CNN.png)
> implemented by: Yiming Wu
---

# Part 1: Reproduction
## 1.4: training 
- Divergent original param 
- Hyperparameter tuning
- Under experimentation

![bg right:54% left: 67% fit vertical](./discussion_img/normal_converge.png)
![bg right:54% left: 67% fit vertical](./discussion_img/origin_param.png)
> implemented by: Team

---
# Part 1: Reproduction
## 1.4: training
- Result comparision
- visualization
- **original**: pearson **0.79**, RMSE **1.33**
- **current best**: pearson **0.72**, RMSE **1.55**

![bg right:54% left: 67% fit vertical](./discussion_img/reproduction_metrics_comparison_seaborn.png)
![bg right:54% left: 50% fit vertical](./discussion_img/reproduction_scatter_seaborn.png)
> implemented by: Yiyang Tan
---
# Part 2: AF3 Result
## 2.1 Generation
- Online server: **Chai Discovery**
- Finished all 285 coreset proteins

![bg right:50% left: 67% fit vertical](./discussion_img/chai.png)
![bg right:50% left: 67% fit vertical](./discussion_img/MSA.png)

> implemented by: Team

---
# Part 2: AF3 Result
## 2.2 Pipeline
- Highest score result extraction: **pred_0.cif**
- Direct **.cif** parse, featurazation
- **Bio.PDB.MMCIFParser**
- Undefined atoms/isotope are classified to **others** 

> implemented by: Guo Yu

---
# Part3: Evaluation
## 3.1 result/visualization
- Our current stage
- CASF Sfcnn vs. AF3 Sfcnn vs. **ground truth**
- AF3 not as good as expected

![bg right:54% left: 67% fit vertical](./discussion_img/top20_heatmap.png)
![bg right:54% left: 67% fit vertical](./discussion_img/gap_histogram.png)

> implemented by: Team
---
# Part3: Evaluation
## 3.2 Next Step
- Keep training for more sccurate model
- Gap distribution analysis
- Identify the protein structures with **'wrong'** prediction using **PDBID**
- Make hypothesis based on AF3 training set, modules etc.

![bg right:54% left: 67% fit vertical](./discussion_img/gap_boxplot.png)
![bg right:54% left: 67% fit vertical](./discussion_img/gap_violinplot.png)

> implemented by: Team
---

# Our Expectation

- Finish training process/AF3 final PLA result (1 week)
- Finish report writing, result analysis and presentation preparing (1 week)


