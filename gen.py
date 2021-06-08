import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

n = 20
driver = (0, 0)
win = (400, 400)
pos = np.random.randint(0, high=win[0], size=(n, 2), dtype=int)
dest = np.random.randint(0, high=win[0], size=(n, 2), dtype=int)

np.savez("data/passengers",
         p_num=n,
         e_win=win,
         p_pos=pos,
         p_dest=dest,
         d_pos=driver)

# vis
# kmeans = KMeans(n_clusters=10, random_state=0).fit(pos).labels_
# fig = px.scatter(x=pos[:, 0], y=pos[:, 1], color=kmeans)
# fig.show()