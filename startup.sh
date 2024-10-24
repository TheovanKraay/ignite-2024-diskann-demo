# startup.sh
#!/bin/bash
pip install -r requirements.txt
streamlit run src/app/cosmos-app.py --server.port=8000 --server.address=0.0.0.0