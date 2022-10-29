# EiTMOS

EiTMOS is an electricity theft monitoring system that uses Quantum Machine Learning algorithms to predict the suspicious consumers in real time.

Quantum Machine Algorithm Used for our use case:
- Quantum Support Vector Machine (QSVM)
- Quantum Kernel Training Algorithm (QKT)
- Quantum Neural Network (Customized one –Called QuClassi)

### Code Structure
<pre>
- etheftQML/
   - app.py                                      # Backend (http://localhost:5050/)
   - templates/                                  # Frontend        
      - index.html                               # Landing page
      - dash.html                                # Dashboard
      -  style.css    
      -  styleDash.css

    - models/                                    
      -  qsvcDuplicateAfterSplit.model            # QSVM
      - QKT389.model                              # QKT
      - QuclassiModel.dat                         # QuClassi
   - data/                                        # contains time series data, dataset used to train and test the models.
   </pre>
