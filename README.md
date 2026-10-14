# MPCoT
The identification and implementation of technology opportunities are critical for fostering innovation and organizational development. However, existing technology opportunity analysis (TOA) research primarily focuses on identifying what constitutes a technology opportunity, often neglecting the why (underlying causal mechanisms) and the how (actionable implementation pathways).This lack of explanatory mechanisms significantly limits the practical utility of TOA methods. To bridge this gap, we propose a novel explainable method for technology opportunity identification based on metapaths and multi-perspective chains of thought (MPCoT). Unlike traditional approaches, MPCoT not only identifies potential technology opportunities but also generates clear, human-understandable explanations for each technology opportunity, thereby revealing the internal decision-making process of the underlying predictive model. MPCoT operates on heterogeneous graph neural networks for TOA. Firstly, the framework extracts the computational graph for prediction instances. Utilizing reparameterization techniques, MPCoT automatically learns approximate discrete masks to identify the critical metapaths. Subsequently, MPCoT employs Retrieval-Augmented Generation to equip large language model (LLM) with domain knowledge and introduces a multi-perspective chain of thought to enhance their analytical capabilities. Ultimately, both the implementation paths and justifications for each technology opportunity are generated. Extensive experiments using a LLM domain dataset demonstrate that MPCoT outperforms all baselines on two fidelity metrics. Through a dual evaluation mechanism combining AI-based scoring with human assessment, supplemented by case studies, we provide quantitative and qualitative validation of MPCoT's effectiveness and practical utility. This work offers research managers deeper insight into the logic and mechanisms behind TOA, enabling more informed decisions and better prioritization of high-potential technology opportunities.
## Requirement
* python 3.8.5
* pytorch==1.12.1+cu116
* dgl==1.0.1+cu116
* scipy==1.10.1
* pandas==1.5.3
* networkx==2.5
* numpy==1.20.3
* sklearn

## Usage
Original dataset:
* https://www.scidb.cn/s/EfYN3u

### TOA model training
```python
python TOA_model/train.py
```
### TOA Prediction
```python
python TOA_model/prediction.py
```
### Generation of metapath explanation
```python
python Explainer/explainer_train.py
```
### Generation of natural language explanation
```python
python Explainer/explanation_generator.py
```
