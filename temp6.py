import subprocess
import os

cur_path = os.path.dirname(os.path.abspath(__file__))
scripts = os.path.join(cur_path, 'scripts')

iimjobs = subprocess.run(['python', os.path.join(scripts, 'iimjobs scraper.py')])
