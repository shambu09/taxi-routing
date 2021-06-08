import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from testServer import Server
from log import Logger
from queue import Queue

Logger.clear()
log = Logger.log

q = Queue()
data = np.load("data/passengers.npz", allow_pickle=True)

n = data["p_num"]
pos = data["p_pos"]
dest = data["p_dest"]
driver = data["d_pos"].reshape((1, 2))

training1 = np.concatenate([pos, dest], axis=1)
training2 = np.concatenate([dest, pos], axis=1)
training = np.concatenate([training1, training2], axis=0)

assert (training.shape[1] == 4)
kmeans = KMeans(n_clusters=10, random_state=0).fit(training)

p_d = np.concatenate([pos, dest], axis=0)
color = np.concatenate([np.zeros(n), np.ones(n)], axis=0)
text = kmeans.labels_
fig = px.scatter(
    x=p_d[:, 0],
    y=p_d[:, 1],
    color=text,
    hover_name=["Start" if i == 0 else "Destination" for i in color])

classes = kmeans.predict(training1).reshape((-1, 1))

log(training1.shape)
log(classes.shape)

p_data = training1

#check driver belongs to which  cluster
driver = np.concatenate([driver, driver], axis=1)
d_cluster = kmeans.predict(driver)
log(d_cluster)

log("\n")
cls = kmeans.cluster_centers_
log(cls)

if __name__ == '__main__':
    app = Server()
    app(fig)
    app.run()