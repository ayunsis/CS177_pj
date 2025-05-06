import pandas as pd
import numpy as np

core2016_df= pd.read_csv('src\AF3_eval\outputs\output_core2016.csv')
sfcnn_df = pd.read_csv('src\AF3_eval\outputs\output_sfcnn.csv')

core2016_df['gap'] = (core2016_df['score'] - core2016_df['affinity']).abs()
sfcnn_df['gap'] = (sfcnn_df['score'] - sfcnn_df['affinity']).abs()
core2016_df['gap'] = core2016_df['gap'].round(2)
sfcnn_df['gap'] = sfcnn_df['gap'].round(2)
core_top10 = core2016_df.nlargest(10, 'gap')
sfcnn_top10 = sfcnn_df.nlargest(10, 'gap')
sfcnn_core_gap = (core2016_df['gap'] - sfcnn_df['gap']).abs()
sfcnn_core_gap = sfcnn_core_gap.round(2)
print(core_top10[['pdbid','score','affinity','gap']])
print('='*50)
print(sfcnn_top10[['pdbid','score','affinity','gap']])
print('='*50)
print(sfcnn_core_gap.nlargest(10))
print('='*50)
print('sfcnn total gap:', sfcnn_df['gap'].sum().round(2))
print('core2016 total gap:', core2016_df['gap'].sum().round(2))
print('sfcnn core total gap:', sfcnn_core_gap.sum())