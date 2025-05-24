The limitations of AlphaFold3 (AF3) in protein-ligand affinity prediction stem from several factors related to its design, training methodology, and architectural focus. Here's a detailed elaboration on these reasons:

### 1. **Lack of explicit energy calculation capabilities**
   - **Diffusion-based generation**: AF3 uses a diffusion-based approach to predict protein-ligand structures. While this method is effective for generating plausible structural conformations, it does not inherently incorporate the physics-based energy calculations necessary for accurately predicting binding affinity. Unlike molecular dynamics simulations or free energy perturbation methods, AF3 does not explicitly model the energetic interactions (e.g., van der Waals forces, electrostatic interactions, hydrogen bonds) that determine binding strength.
   - **No explicit representation of binding energetics**: AF3's architecture focuses on structural plausibility rather than energetic accuracy. It lacks dedicated modules or parameters to quantify the thermodynamic contributions of specific interactions, which are critical for affinity prediction.

### 2. **Training data limitations**
   - **Underrepresentation of binding affinity data**: AF3's training data primarily consist of experimentally determined protein structures (e.g., from the PDB) and does not include extensive datasets of binding affinities for diverse protein-ligand pairs. Without explicit training on binding affinity data, the model cannot learn the nuanced relationships between structural features and binding strength.
   - **Limited conformational diversity**: The training data predominantly consist of static crystallographic structures, which may not adequately represent the dynamic conformational changes and ensemble states that occur during ligand binding. This limitation affects AF3's ability to model the subtle structural adjustments that influence binding affinity.

### 3. **Model architecture constraints**
   - **Focus on structural accuracy over energetic nuance**: AF3 is optimized to predict the 3D coordinates of atoms in protein-ligand complexes with high precision. However, its architecture does not prioritize the quantitative prediction of binding free energy or affinity. The model's loss function and training objectives emphasize structural alignment rather than energetic accuracy.
   - **No explicit scoring function for affinity**: While AF3 provides metrics like ranking scores and predicted aligned error (PAE), these are not designed to correlate with binding affinity. They primarily assess the confidence in the predicted structure rather than the strength of the interaction.

### 4. **Challenges in modeling dynamic systems**
   - **Static vs. dynamic complexes**: AF3 performs well for static protein-ligand complexes (minimal conformational change) but struggles with dynamic systems involving significant induced fit or conformational rearrangements. Binding affinity in such systems often depends on the protein's ability to adapt its structure upon ligand binding, which AF3 may not capture accurately.
   - **GPCR conformational bias**: The model exhibits a systematic bias toward predicting active conformations of GPCRs, even for antagonists that stabilize inactive states. This suggests that AF3 may not adequately model the energy landscape required to distinguish between different functional states, which are critical for accurate affinity predictions in such systems.

### 5. **Data memorization vs. generalizable learning**
   - **Memorization of training data**: AF3's performance declines significantly on structures released after its training cutoff date. This indicates that the model may rely on memorizing patterns in its training data rather than learning generalizable principles of molecular recognition. As a result, it may not reliably predict binding affinity for novel protein-ligand pairs outside its training distribution.

### 6. **Difficulty in modeling ternary complexes**
   - **Complex interaction networks**: Ternary complexes (e.g., molecular glues and PROTACs) involve intricate protein-protein and protein-ligand interactions. AF3's current implementation struggles to accurately model these complex systems, which are often critical for determining binding affinity in therapeutic applications.
   - **Missing specific interaction patterns**: Binding affinity in ternary systems often depends on the precise arrangement of multiple interactions (e.g., hydrogen bonds, π-stacking, electrostatic complementarity). AF3 may fail to capture these subtle but critical determinants of binding strength.

### 7. **Kinome selectivity prediction challenges**
   - **Subtle determinants of selectivity**: Kinase selectivity profiles depend on the absence of specific favorable interactions rather than outright steric clashes. AF3 appears better at identifying structural incompatibilities (negative determinants) than recognizing the specific interaction patterns (positive determinants) required for binding. This asymmetry limits its ability to predict affinity across highly similar kinase targets.

### Summary
AF3's limitations in protein-ligand affinity prediction arise from its architectural focus on structural plausibility rather than energetic accuracy, training data biases, and challenges in modeling dynamic and complex systems. While it excels at generating high-quality structural models for known binding pairs with minimal conformational change, it lacks the necessary features to reliably rank compounds by binding affinity. Addressing these limitations would require integrating physics-based energy calculations, enhancing conformational sampling, and enriching training data with diverse binding affinity information.