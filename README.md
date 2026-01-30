Objwct Verification and Livelyness

python3.13 -m venv .venv-3-13-2
OR
python3.10 -m venv .venv-3-10-11
source .venv-3-13-2/bin/activate



brew install ngrok



Now back to Project:

pip install -r requirements.txt

Activate environment:
source .venv/bin/activate

Run project:
python test-api.py
OR

uvicorn app:main_app --host 0.0.0.0 --port 8000 --workers 1




To make it online run cloudflade Tunnel after running local Server:
(Use Port you want to expose online):
cloudflared tunnel --url http://localhost:5000

Or "ngrok" and config or Auth for the first time
ngrok config add-authtoken <Auth-Token>

Run using key
ngrok http <PortNumber/5000>

ngrok http 8000
OR
ngrok http -region eu <PortNumber/5000>
OR
ngrok http --domain=your-subdomain-name.ngrok-free.app 5000   
#fusionsuite-io-vehicle-test