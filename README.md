# EiTMOS

EiTMOS is an electricity theft monitoring system that uses Quantum Machine Learning algorithms to predict the suspicious consumers in real time.

Quantum Machine Algorithm Used for our use case:
- Quantum Support Vector Machine
- Quantum Kernel Training Algorithm
- Quantum Neural Network (Customized one –Called QuClassi)

< PROJECT ROOT >
   |
   |-- etheftQML/
   |    |
   |    |     
   |    |-- app.py                                        # Backend (http://localhost:5050/)
   |    |
   |    |-- templates/                                    # Frontend        
   |    |    |-- index.html                               # Landing page
   |    |    |-- dash.html                                # Dashboard
   |    |    |-- style.css    
   |    |    |-- styleDash.css
   |    |    |
   |    |-- models/                                       # UI Kit Pages
   |    |    |-- qsvcDuplicateAfterSplit.model            # Index page
   |    |    |-- QKT389.model                             # 404 page
   |    |    |-- QuclassiModel.dat                        # All other pages
   |    |-- data/                                         # contains time series data, dataset used to train and test the models.
   |-- ************************************************************************
