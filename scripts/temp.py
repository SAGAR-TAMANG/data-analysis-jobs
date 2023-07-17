import pandas as pd
import os
import datetime

dff101 = pd.DataFrame([[101, 202]], columns=['hey', 'hello'])

dff101.to_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', "fake" + str(datetime.date.today()) + ".xlsx"), index=False)
