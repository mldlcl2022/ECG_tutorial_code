# ECG tutorial code
Chan Lee, Seokhyeon Ha  

### Installation
```bash
pip install numpy==1.26.4
pip install torch==1.13.1

pip install wfdb
pip install pandas==2.2.3

pip install tqdm
pip install scikit-learn==1.5.2
pip install matplotlib
```

### Training
```python
!python main.py
```

### Results
Epoch 29: 100%|██████████████████████████████| 67/67 [00:11<00:00,  6.08batch/s]
Time: 0m 18s Loss: 0.2698

Epoch 30: 100%|██████████████████████████████| 67/67 [00:10<00:00,  6.19batch/s]
Time: 0m 18s Loss: 0.2692
Train | AUC: 0.9371, F1: 0.7137
Valid | AUC: 0.9108, F1: 0.6427

Result of testset
AUC: 0.9054, F1-score: 0.6782

### Contact
Should you have any questions or concerns, please feel free to contact me at the email below. ⬇️</br>
<div align="center"> 📫 mldlcl2022@gmail.com 📫 </div>

### Acknowledgements
Thank you for our [Lab](https://www.k-medai.com/home).  
If you find this code useful, please consider citing our work.
