import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from log import Logger
from queue import Queue
import plotly.graph_objects as go

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
fig = px.scatter(x=p_d[:, 0],
                 y=p_d[:, 1],
                 color=color,
                 hover_name=text,
                 title="Taxi Routing Based on Passenger Destination")

classes = kmeans.predict(training1).reshape((-1, 1))

log(training1.shape)
log(classes.shape)

curr_center = np.concatenate(
    (kmeans.cluster_centers_.squeeze(), np.arange(0, 10).reshape(
        10, 1), np.ones((10, 1))),
    axis=1)

curr_passengers = np.concatenate((training1, np.ones((training1.shape[0], 1))),
                                 axis=1)


#*------------------------------Helper funcs-------------------------------------#
def getSameClass(_class, classes, data):
    """
    *get passengers with same class
    @param _class  -> int
    @param classes -> vector :: (num_samples, 1)
    @param data    -> vector :: (num_samples, 5)
    """
    _class = int(_class)
    assert (isinstance(_class, int))
    assert (classes.shape[1] == 1)
    assert (data.shape[1] == 5)

    mask = classes == _class
    return data[mask.squeeze()]


def getNearestCentreClass(centers, driver):
    """
    *get nearest center and remove it from the list
    @param centers -> vector :: (10, 6)
    @param driver  -> vector :: (1, 2)
    """
    driver = driver.reshape((1, 2))
    assert (centers.shape[1] == 6)
    assert (driver.shape == (1, 2))

    mask = centers[:, -1] == 1
    tmp = centers[mask]
    assert (tmp.shape[1] == centers.shape[1])

    dis = np.linalg.norm(tmp[:, 0:2].copy() - driver)
    mini = np.argmin(dis)
    _class = tmp[mini, 4]
    tmp[mini, 5] = 0
    centers[mask] = tmp
    return _class, centers


def getNearestPassenger(driver, passengers):
    """
    *get nearest passenger in the same class
    @param driver      -> vector :: (1, 2)
    @param passengers  -> vector :: (nums_samples, 5)
    """
    driver = driver.reshape((1, 2))
    assert (driver.shape == (1, 2))
    assert (passengers.shape[1] == 5)

    mask = passengers[:, -1] == 1
    tmp = passengers[mask]

    dis = np.linalg.norm(tmp[:, 0:2].copy() - driver)
    mini = np.argmin(dis)
    tmp[mini, -1] = 0
    passengers[mask] = tmp
    return tmp[mini, 0:4], passengers, tmp.shape[0] - 1


#*------------------------------------------------------------------------------#

"""
#!-------------------------------taxi starting----------------------------------#
#check driver belongs to which  cluster
driver = np.array([0, 0])
assert (driver.shape == curr_passengers[0, 0:2].shape)

queue = []
tmp_passengers = curr_passengers
k = 1
traversed = 0
f = 0

while (traversed < 10):
    f += 1
    if f == 1000:
        print("Stack limit")
        break

    d_class, curr_center = getNearestCentreClass(curr_center, driver)
    tmp_passengers = getSameClass(d_class, classes, curr_passengers)

    while (k != 0):
        tmp_driver, tmp_passengers, k = getNearestPassenger(
            driver, tmp_passengers)
        queue.append((tmp_driver[0:2], tmp_driver[2:]))
        driver = tmp_driver[0:2]

    k = 1
    traversed += 1
np.save("data/queue", queue)
#!-------------------------------------------------------------------------------#
"""

q = np.load("data/queue.npy")
q = q.reshape((-1, 4))
x = []
y = []
for i in range(20):
    x.append(q[i, 0])
    y.append(q[i, 1])
    x.append(q[i, 2])
    y.append(q[i, 3])

tmp = [0]
tmp2 = [0]


def append(tmp, el):
    tmp.append(el)
    return tmp


frames = [
    go.Frame(data=[go.Scatter(x=append(tmp, x[i]), y=append(tmp2, y[i]))])
    for i in range(40)
]

fig2 = go.Figure(
    data=[go.Scatter(x=[0], y=[0], line=dict(color="rgb(189,189,189)"))],
    layout=go.Layout(
        xaxis=dict(range=[0, 400], autorange=False),
        yaxis=dict(range=[0, 400], autorange=False),
        title="Taxi Routing Based on Passenger Destination",
        updatemenus=[
            dict(type="buttons",
                 buttons=[dict(label="Play", method="animate", args=[None])])
        ]),
    frames=frames)


fig.update_traces(marker=dict(size=6,
                              line=dict(width=1,
                                        color='black')),
                  selector=dict(mode='markers'))

fig.update_traces(marker_coloraxis=None)
fig2.add_trace(fig.data[0])
fig3 = go.Figure(
    data=go.Scatter(x=x, y=y, line=dict(color="rgb(189,189,189)")),
    layout=go.Layout(xaxis=dict(range=[-10, 400], autorange=False),
                     yaxis=dict(range=[-10, 400], autorange=False),
                     title="Taxi Routing Based on Passenger Destination"))

fig3.update_layout(xaxis_title="x", yaxis_title="y")
fig3.add_trace(fig.data[0])
fig3.update_traces(showlegend=False)

fig2.update_traces(showlegend=False)
log(q)

if __name__ == '__main__':
    fig3.write_image("animation/final.png")
    fig.write_image("animation/plot.png")
    fig2.show()