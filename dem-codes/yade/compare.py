import sys
import subprocess

sys.path.append('../../functions')

from analyse_functions import *
from helpers import *

# this will write myMotion.out
subprocess.run(['yadedaily','-nx', 'generateForces.py','myMotion.txt'], check=True)

ref = json_to_dict("theoreticalResult.json")
dem = load_grouped_csv("myMotion.out")
report = my_compare_results(dem, ref)
print(report)