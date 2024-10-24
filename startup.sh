# startup.sh
#!/bin/bash
pip install -r requirements.txt
streamlit run src/app/cosmos-app.py --server.port=443 --server.address=0.0.0.0